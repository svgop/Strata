#!/usr/bin/env python3
"""probe-m1-anatomy.py - where the M1 number's wall time goes: start latency or decode rate.

The lane's M1 (`campaign/probe-lane.py`) is one chat request scored as tokens / full wall time, so it
folds queueing, prefill and response overhead into a number everyone reads as decode speed. The config
ladder of 2026-11-24 found every boot's third leg 4-9% below its second with the decode window cost
IDENTICAL (13.0 ms/window) and acceptance HIGHER on the slow leg - the deficit sat before the first
token. This instrument separates the terms on the standing recipe, in one boot:

    S S   two lone M1 requests (the quiet reference)
    C     a conc burst shaped like the probe's 6-conc lane (6 x 600 sampled tokens)
    S S   the same M1 request at once, then once more (what the next leg actually meets)
    C S S again (the pattern repeats or it was noise)

Per request: wall, TTFT (first streamed delta - content or reasoning), decode rate over the streamed
tokens, and the engine's own line for that request (ms/window, accept, hit) from the batch accounting.
Boot/teardown come from campaign/config-abab.py (the standing launcher's env arms, battery-lib's kill).

    python campaign/probe-m1-anatomy.py --label m1-anatomy [--pin-profile FILE] [--config CONFIG]
"""
import argparse, importlib.util, json, pathlib, re, shutil, subprocess, sys, threading, time, urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cab", REPO / "campaign/config-abab.py")
cab = importlib.util.module_from_spec(spec); spec.loader.exec_module(cab)
KEY = cab.KEY
M1_PROMPT = ('Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on '
             'LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and '
             'pinned memory tradeoffs, with concrete numbers and examples throughout.')   # probe-lane.py REAL
CONC_PROMPT = ('Explain in detail how mixture-of-experts routing works in modern LLMs, including load balancing, '
               'capacity factors, and the tradeoffs versus dense models.')                         # probe-lane.py CONC



def stream(prompt: str, max_tok: int, temp: float) -> dict:
    """One streamed chat request; returns wall, ttft, streamed token count and the decode rate."""
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tok, "temperature": temp, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); ttft = None; chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): usage = d["usage"]
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.time() - t0
            chunks += 1
    wall = time.time() - t0
    tok = (usage or {}).get("completion_tokens") or chunks
    return {"wall": wall, "ttft": ttft or wall, "tok": tok,
            "decode_tps": tok / max(1e-6, wall - (ttft or 0.0)), "lane_tps": tok / wall}


def accounting_tail(n: int = 3) -> list:
    """The engine's own per-request lines (ms/window | accept | hit), most recent last."""
    log = REPO / "logs/custom-iq3_s.log"
    with open(log, "rb") as fh:
        fh.seek(max(0, log.stat().st_size - 400_000)); tail = fh.read().decode("utf-8", "replace")
    return re.findall(r"prompt\s+\d+ gen\s+\d+ ->\s+([\d.]+|-) ms/window \| accept\s+([\d.]+)%.*?hit\s+([\d.]+)%", tail)[-n:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="m1-anatomy"); ap.add_argument("--config", default="config/custom-iq3_s-8081.json")
    ap.add_argument("--pin-profile")
    ap.add_argument("--temp", type=float, default=0.7,
                    help="0 = greedy: every M1 request generates IDENTICAL text, so a post-burst slowdown is engine "
                         "state and not sampled content (the drafter's confidence is content-dependent at 0.7)")
    ap.add_argument("--burst-prompt", choices=("conc", "same"), default="conc",
                    help="conc = the burst routes like the probe's CONC lane (a signature CHANGE for the returning M1); "
                         "same = the burst uses the M1 prompt itself (same load, same duration, NO signature change)")
    ap.add_argument("--burst", choices=("conc", "single"), default="conc",
                    help="conc = the lane's 6-conc shape; single = ONE lone request of equal duration (same heat, "
                         "different traffic shape: the differential isolates the adaptive tier from thermals)")
    a = ap.parse_args()
    out = REPO / "campaign/results" / a.label; out.mkdir(parents=True, exist_ok=True)
    profile = REPO / "data/expert-profile-learned.bin"
    if a.pin_profile: shutil.copyfile(a.pin_profile, profile)
    cab.cleanup(); cab.boot(REPO / a.config, out, "anatomy")
    stream(M1_PROMPT, 200, a.temp)                      # first-touch discard (F-360)
    rows = []
    def lone(tag):
        m = stream(M1_PROMPT, 2000, a.temp); m.update(tag=tag, kind="M1"); rows.append(m)
        acc = accounting_tail(1)
        print(f"  {tag:4s} M1  wall {m['wall']:.2f}s  ttft {m['ttft']:.3f}s  tok {m['tok']}  decode {m['decode_tps']:.1f} t/s  lane-number {m['lane_tps']:.1f}  engine {acc[-1] if acc else '?'}", flush=True)
    gpu = []; stop = threading.Event()
    def sample_gpu():
        """SM clock, temperature and power once a second - the thermal arm's own evidence."""
        while not stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            try:
                sm, temp, pw = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
                gpu.append({"t": time.time(), "sm": sm, "temp": temp, "pw": pw})
            except (ValueError, IndexError):
                pass
            stop.wait(1.0)
    sampler = threading.Thread(target=sample_gpu, daemon=True); sampler.start()
    marks = {}
    def burst(tag):
        marks[tag] = time.time()
        bp = M1_PROMPT if a.burst_prompt == "same" else CONC_PROMPT
        if a.burst == "single":                       # same duration, single-stream shape
            m = stream(bp, 3600, a.temp)
            print(f"  {tag:4s} single-stream burst: {m['tok']} tok in {m['wall']:.2f}s", flush=True); return
        res = [None] * 6
        def one(i): res[i] = stream(bp, 600, a.temp)
        ts = [threading.Thread(target=one, args=(i,)) for i in range(6)]
        t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
        print(f"  {tag:4s} 6-conc burst: {sum(r['tok'] for r in res)} tok in {time.time() - t0:.2f}s", flush=True)
    lone("S1"); lone("S2"); marks["C1"] = time.time(); burst("C1"); marks["S3"] = time.time(); lone("S3"); lone("S4")
    burst("C2"); marks["S5"] = time.time(); lone("S5"); lone("S6")
    stop.set(); sampler.join(timeout=3)
    (out / "rows.json").write_text(json.dumps(rows, indent=1)); (out / "gpu.json").write_text(json.dumps({"marks": marks, "samples": gpu}, indent=1))
    cab.cleanup()
    t3 = marks["S3"]; during = [g for g in gpu if t3 - 15 <= g["t"] <= t3 - 1]; after = [g for g in gpu if t3 + 1 <= g["t"] <= t3 + 8]
    if during and after:
        m_ = lambda xs, k: sum(x[k] for x in xs) / len(xs)
        print(f"  GPU around S3: burst SM {m_(during, 'sm'):.0f} MHz {m_(during, 'temp'):.0f}C {m_(during, 'pw'):.0f}W -> "
              f"S3 SM {m_(after, 'sm'):.0f} MHz {m_(after, 'temp'):.0f}C {m_(after, 'pw'):.0f}W")
    lone_mean = lambda keys, f: sum(r[f] for r in rows if r["tag"] in keys) / len(keys)
    quiet, after = ("S1", "S2", "S4", "S6"), ("S3", "S5")
    print(f"\n=== M1 ANATOMY [{a.label}]  (S3/S5 = the first lone request after a conc burst)")
    for name, keys in (("lone/quiet", quiet), ("after-conc", after)):
        print(f"  {name:11s} ttft {lone_mean(keys, 'ttft'):.3f}s  decode {lone_mean(keys, 'decode_tps'):.1f} t/s  lane-number {lone_mean(keys, 'lane_tps'):.1f} t/s")


if __name__ == "__main__":
    main()
