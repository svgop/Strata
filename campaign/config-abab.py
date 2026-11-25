#!/usr/bin/env python3
"""config-abab.py - THE CONFIG LADDER: is one config variable costing the lanes?

The binary ladder (campaign/binary-ladder.sh) answers that question for an engine build. This is its
counterpart for a CONFIG variable on one binary - same boot path as the standing launcher (the lib's
`launch::engine_argv` under the launcher's own env arms, campaign/lib/launch-common.sh), two configs,
A/B alternating across boots in one window (law 17), the learned profile pinned to one file before
EVERY boot (F-271: the standing saves it, so an unpinned run compares profile states, not configs),
and the warm protocol paid per boot (the server's prewarm, then leg 1 discarded - law 11 / F-360).

    python campaign/config-abab.py --label m1-phase --config-a config/a.json --config-b config/b.json \
        --pin-profile data/expert-profile-learned.pinned.bin [--boots 4] [--legs 3]

Verdict = per-arm MEANS over the measured legs (2..N) of every boot, A against B on M1, 6-conc and
M3b; the band is the campaign's +-4%. Teardown is battery-lib's `cleanup_engine` (the one kill owner).
Full probe output is kept per leg under campaign/results/<label>/; the server is left DOWN.
"""
import argparse, atexit, json, os, pathlib, re, shutil, subprocess, sys, time, urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "launch-custom-iq3s-8081.sh"
RESULTS = REPO / "campaign/results"
KEY = json.loads((REPO / "config/custom-iq3_s-8081.json").read_text())["api_key"]
ANCHOR_RE = re.compile(r"^launch::exec_engine --config \S+ --port 8081 --host 0\.0\.0\.0$")
# the server's validated steering handshake (serve/server.py prints it from engine.info["cvec"] once the gate has
# checked the armed tables against the config).  /health does NOT carry it: its "engine" field is the version string.
ARMED_RE = re.compile(r"^\[strata\] control vector: armed (\S+) \(validated against the config\)", re.M)

_proc = None


def cleanup():
    global _proc
    subprocess.run(["bash", "-c", f". {REPO}/campaign/battery-lib.sh; cleanup_engine"], cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if _proc is not None:
        try:
            _proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    _proc = None


atexit.register(cleanup)


def health():
    try:
        req = urllib.request.Request("http://127.0.0.1:8081/health", headers={"Authorization": f"Bearer {KEY}"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read())
    except Exception:
        return {}


def armed_geometry(log: pathlib.Path, pos: int) -> str:
    """The validated steering handshake this boot printed to `log` after byte offset `pos`; "?" when there is none."""
    try:
        return ARMED_RE.findall(log.read_text(errors="replace")[pos:])[-1]
    except IndexError:
        return "?"


def boot(config: pathlib.Path, out: pathlib.Path, tag: str) -> str:
    """Boot the standing recipe under the launcher's own env arms with `config`; return the armed geometry."""
    global _proc
    last = LAUNCHER.read_text().rstrip("\n").splitlines()[-1].strip()
    if not ANCHOR_RE.match(last):
        sys.exit(f"REFUSING: the launcher's last line is no longer the exec anchor: {last!r}")
    argv_file, env_file = out / f"{tag}.argv", out / f"{tag}.env0"
    script = (f'source <(sed "$ d" {LAUNCHER.name}) && launch::engine_argv --config {config} --port 8081 '
              f'--host 127.0.0.1 > "{argv_file}" && env -0 > "{env_file}"')
    with open(out / "launcher.log", "a") as fh:
        r = subprocess.run(["bash", "-c", script], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not argv_file.exists():
        sys.exit(f"REFUSING: the launcher exited {r.returncode} before its exec line; see {out / 'launcher.log'}")
    argv = argv_file.read_text().split()
    if argv[:4] != [".venv/bin/python", "serve/server.py", "--engine", "strata"] or str(config) not in argv:
        sys.exit(f"REFUSING: unexpected engine argv {argv}")
    argv[0] = str(REPO / ".venv/bin/python")
    env = dict(i.split("=", 1) for i in env_file.read_bytes().decode("utf-8", "surrogateescape").split("\0") if i)
    arms_vector = "--control-vector-scaled" in json.loads(config.read_text()).get("args", [])
    log = out / "server.log"
    pos = log.stat().st_size if log.exists() else 0
    with open(log, "a") as fh:
        fh.write(f"\n===== {tag} {' '.join(argv)} {time.strftime('%F %T')} =====\n"); fh.flush()
        _proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        h = health()
        if h.get("prewarm") in ("done", "failed", "yielded"):
            cvec = armed_geometry(log, pos)
            if cvec == "?" and arms_vector:
                print(f"  [{tag}] WARNING: the config arms a control vector and the server printed no validated "
                      f"'control vector: armed' line - this boot's steering geometry is UNVERIFIED", flush=True)
            print(f"  [{tag}] up in {time.time() - t0:.0f}s prewarm={h['prewarm']} cvec={cvec}", flush=True)
            return cvec
        if _proc.poll() is not None:
            sys.exit(f"REFUSING: engine exited early ({_proc.returncode}); see {out / 'server.log'}")
        time.sleep(4)
    sys.exit(f"REFUSING: no prewarm word within 900s; see {out / 'server.log'}")


def leg(label: str, keep: pathlib.Path) -> dict:
    r = subprocess.run([str(REPO / ".venv/bin/python"), "campaign/probe-lane.py", "--key", KEY, "--label", label],
                       cwd=REPO, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    keep.write_text(txt)
    g = lambda p: (re.search(p, txt) or [None, None])[1]
    return {"m1": float(g(r"M1 single-stream : ([\d.]+) t/s") or "nan"),
            "conc": float(g(r"M2 6-conc\s*: ([\d.]+) t/s") or "nan"),
            "m3b": float(g(r"M3b rate\s*: ([\d.]+)%") or "nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True); ap.add_argument("--config-a", required=True); ap.add_argument("--config-b", required=True)
    ap.add_argument("--boots", type=int, default=4); ap.add_argument("--legs", type=int, default=3)
    ap.add_argument("--pin-profile", help="restore this file to data/expert-profile-learned.bin before every boot")
    a = ap.parse_args()
    out = RESULTS / a.label; out.mkdir(parents=True, exist_ok=True)
    profile = REPO / "data/expert-profile-learned.bin"
    ca, cb = (pathlib.Path(x) for x in (a.config_a, a.config_b))
    for c in (ca, cb):
        if not (c if c.is_absolute() else REPO / c).exists():
            sys.exit(f"REFUSING: {c} does not exist")
    rows = []
    for k in range(a.boots):
        arm = "A" if k % 2 == 0 else "B"; cfg = ca if arm == "A" else cb
        if a.pin_profile:
            shutil.copyfile(a.pin_profile, profile)
        cleanup()
        cvec = boot(cfg if cfg.is_absolute() else REPO / cfg, out, f"{arm}{k}")
        for i in range(1, a.legs + 1):
            m = leg(f"{a.label}-{arm}{k}-leg{i}", out / f"{arm}{k}-leg{i}.txt")
            m.update(arm=arm, boot=k, leg=i, measured=i >= 2, cvec=cvec, load=os.getloadavg()[0])
            rows.append(m)
            print(f"  [{arm}{k} leg{i}{'  (discard)' if i == 1 else ''}] M1 {m['m1']:.1f}  conc {m['conc']:.1f}  "
                  f"M3b {m['m3b']:.1f}%  load {m['load']:.2f}", flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
    cleanup()
    mean = lambda arm, key: (lambda v: sum(v) / len(v) if v else float("nan"))([r[key] for r in rows if r["arm"] == arm and r["measured"]])
    print(f"\n=== CONFIG LADDER [{a.label}]  measured legs only, means over {a.boots} boots x {a.legs - 1} legs")
    print(f"{'lane':10s} {'A':>9s} {'B':>9s} {'A vs B':>9s}   band +-4%")
    for key, name in (("m1", "M1"), ("conc", "6-conc"), ("m3b", "M3b %")):
        x, y = mean("A", key), mean("B", key)
        print(f"{name:10s} {x:9.1f} {y:9.1f} {100 * (x - y) / y:+8.1f}%")
    print(f"A = {ca}  |  B = {cb}")


if __name__ == "__main__":
    main()
