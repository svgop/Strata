#!/usr/bin/env python3
"""t50-131k-probe.py - T50: warmed 131K solo + concurrent depth probe (A/B reusable).

Turn 1 warms each lane: a ~--depth-token deterministic filler prompt (cold prefill,
conversation parked). Turn 2 continues the SAME conversation (conversation-cache
recovery, then steady decode at depth). Solo = 1 lane; conc = N lanes in parallel
(the batch lane, batch-mtp D=1). Client SSE cadence + engine /metrics window stats.

Usage: python t50-131k-probe.py --label base --lanes solo,conc --depth 131072 --decode 512
Writes campaign/results/t50-131k-<label>.json and prints headline lines.
"""
import argparse, json, random, threading, time, urllib.request, os

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8081)
ap.add_argument('--label', default='run')
ap.add_argument('--lanes', default='solo,conc')
ap.add_argument('--depth', type=int, default=131072)
ap.add_argument('--decode', type=int, default=512)
ap.add_argument('--conc', type=int, default=6)
ap.add_argument('--key', default='3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8')
ap.add_argument('--temp', type=float, default=0.7)
a = ap.parse_args()

URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"
HEAD = {"Content-Type": "application/json", "Authorization": "Bearer " + a.key}

# Deterministic pseudo-text (depth-lane.py's SECTION pattern, per-lane seeds)
WORDS = ('system atlas vector module harbor quantum ledger freight meadow cipher lantern '
         'orchid tunnel granite beacon piston willow saffron relay cobalt theorem pasture '
         'velocity dossier crescent furnace lattice omens kettle prairie spline terrace '
         'umbra vellum wicket zephyr anvil marigold socket trellis basalt krill napkin '
         'onyx pivot quiver rune siphon tundra urn vortex wharf yield zenith').split()
SECTION = ('Record {n}: field agents surveyed the {w1} installation near the {w2} perimeter. '
           'The {w3} telemetry showed nominal drift across {n} calibrated spans; the {w4} '
           'ledger reconciled with the {w5} archive within tolerance. Follow-up: inspect the '
           '{w6} conduit at the next {w7} cycle and file the {w8} annex.\n')
ASK1 = '\n\nAfter reading the records above, write a 120-word summary of what the field agents will do next.'
ASK2 = ('Now write a detailed 600-word technical essay about estuarine tide-pool ecosystems, '
        'covering species, tides, and conservation, ending with a bulleted summary.')


def build_filler(records, seed):
    rng = random.Random(seed)
    parts = []
    for i in range(records):
        parts.append(SECTION.format(n=i + 1, **{f'w{j}': rng.choice(WORDS) for j in range(1, 9)}))
    return ' '.join(parts)


def post(body, timeout=7200):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HEAD)
    return urllib.request.urlopen(req, timeout=timeout)


def turn1(prompt):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt + ASK1}],
            "temperature": a.temp, "max_tokens": 96, "stream": False}
    t0 = time.time()
    with post(body) as r:
        out = json.load(r)
    wall = time.time() - t0
    u = out.get("usage", {})
    txt = "".join((c.get("message", {}).get("content") or c.get("message", {}).get("reasoning_content") or "")
                  for c in out.get("choices", []))
    txt = txt or out.get("choices", [{}])[0].get("text", "")
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), wall, out.get("timings", {}), txt


def turn2(prompt, reply1, out):
    body = {"model": "m", "messages": [
                {"role": "user", "content": prompt + ASK1},
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": ASK2}],
            "temperature": a.temp, "max_tokens": a.decode, "stream": True}
    t0 = time.time()
    ttft = None
    t_last = 0.0
    chunks = 0
    usage = {}
    with post(body) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                continue
            try:
                obj = json.loads(p)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    t_last = time.time() - t0
    ctok = usage.get("completion_tokens") or chunks
    dec = (t_last - ttft) if ttft is not None and t_last > ttft else float("nan")
    out.update({"ttft_s": round(ttft, 3) if ttft is not None else None,
                "decode_s": round(dec, 3) if dec == dec else None,
                "ctok": ctok,
                "decode_tps": round(ctok / dec, 1) if dec == dec and dec > 0 else None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "recovered_hint": usage.get("conversation_cache_tokens",
                                            usage.get("cached_tokens", None))})
    return out


def metrics():
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/metrics", headers=HEAD)
        with urllib.request.urlopen(req, timeout=10) as r:
            m = json.load(r)
        w = m.get("window", {})
        return {"window_p50_ms": w.get("p50_ms"), "window_p99_ms": w.get("p99_ms"),
                "rows_per_window": w.get("rows_per_window"),
                "requests": m.get("requests", [])[:8]}
    except Exception as e:
        return {"error": str(e)}


def calibrate():
    body = {"model": "m", "messages": [{"role": "user", "content": build_filler(300, 999) + ASK1}],
            "temperature": a.temp, "max_tokens": 1, "stream": False}
    with post(body) as r:
        out = json.load(r)
    pt = out.get("usage", {}).get("prompt_tokens", 420)
    return pt / 300.0


def run_lane(lane_idx, records, seed, res):
    prompt = build_filler(records, seed)
    pt1, ct1, w1, tm1, reply1 = turn1(prompt)
    res["reply"] = reply1
    res["turn1"] = {"prompt_tokens": pt1, "completion_tokens": ct1, "wall_s": round(w1, 2),
                    "prefill_tps": round(pt1 / w1, 0) if w1 else None,
                    "server_timings": {k: round(v, 3) if isinstance(v, (int, float)) else v
                                       for k, v in (tm1 or {}).items()}}
    turn2(prompt, reply1, res)


def main():
    tpr = calibrate()
    records = max(1, int((a.depth - 200) / tpr))
    print(f"# t50 calibration: {tpr:.2f} tok/record -> {records} records (target {a.depth})", flush=True)
    results = {"label": a.label, "depth_target": a.depth, "decode_max": a.decode, "lanes": {}}
    for mode in [x.strip() for x in a.lanes.split(',')]:
        n = 1 if mode == 'solo' else a.conc
        lanes = []
        t_setup0 = time.time()
        # sequential turn-1 (clean cold prefills, parks each conversation)
        for i in range(n):
            res = {}
            run_lane(i, records, 4242 + i * 1009, res)
            lanes.append(res)
            print(f"# [{mode}] lane {i+1}/{n} warmed: prefill {res['turn1']['prefill_tps']:.0f} t/s, "
                  f"depth {res['turn1']['prompt_tokens']}", flush=True)
        setup_s = time.time() - t_setup0
        time.sleep(2)  # WSTAT idle flush
        m_before = metrics()
        # parallel turn-2 (the measured leg)
        t0 = time.time()
        ths = []
        for i in range(n):
            lanes[i]["turn2"] = {}
            # rebuild the same prompt (same seed) for turn 2
            th = threading.Thread(target=run_turn2, args=(records, 4242 + i * 1009,
                                     lanes[i]["reply"], lanes[i]["turn2"]))
            th.start(); ths.append(th)
        for th in ths:
            th.join()
        wall = time.time() - t0
        time.sleep(2)  # WSTAT idle flush
        m_after = metrics()
        for l in lanes:
            l.pop("reply", None)
            l["turn1"].pop("reply", None)
        agg = None
        ttfts = [l["turn2"]["ttft_s"] for l in lanes if l["turn2"].get("ttft_s") is not None]
        lasts = [l["turn2"]["decode_s"] + l["turn2"]["ttft_s"] for l in lanes
                 if l["turn2"].get("decode_s") is not None]
        if ttfts and lasts:
            span = max(lasts) - min(ttfts)
            tot = sum(l["turn2"]["ctok"] for l in lanes)
            agg = round(tot / span, 1) if span > 0 else None
        solo_tps = lanes[0]["turn2"].get("decode_tps") if n == 1 else None
        per_lane = [l["turn2"].get("decode_tps") for l in lanes]
        results["lanes"][mode] = {"n": n, "setup_s": round(setup_s, 1), "turn2_wall_s": round(wall, 1),
                                  "aggregate_tps": agg, "solo_tps": solo_tps,
                                  "per_lane_tps": per_lane,
                                  "metrics_before": m_before, "metrics_after": m_after,
                                  "lane_detail": lanes}
        print(f"T50 {a.label} {mode:4s} | depth {lanes[0]['turn1']['prompt_tokens']:6d} | "
              f"{'solo ' if n==1 else 'agg  '}{(solo_tps or agg)!s:>6} t/s | per-lane {per_lane} | "
              f"ttft {ttfts[:1]} s", flush=True)
    os.makedirs("campaign/results", exist_ok=True)
    path = f"campaign/results/t50-131k-{a.label}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"# wrote {path}", flush=True)


def run_turn2(records, seed, reply1, out):
    prompt = build_filler(records, seed)
    turn2(prompt, reply1, out)


if __name__ == "__main__":
    main()
