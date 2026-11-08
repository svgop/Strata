#!/usr/bin/env python3
"""tools/strata-doctor.py - the memory-architecture doctor + per-device pool sizer.

Gold-standard shapes, per the research digests (2026-11-08-firstrun-ux.md,
2026-11-08-pool-sizing-tools.md): brew-style per-check invocation, react-native-style --fix with a
paste-block fallback (never self-sudo), DPDK's read-back-or-fail arming, PostgreSQL's try/on/off
vocabulary, and a one-time nag law (a ledger of verdict hashes: an explainer prints only when the
bad state is NEW; a fix purges it; --json always tells the truth).

Checks:
  C1 pool        armed size vs the catalog requirement (sized by the formula below)
  C2 page-class  the 2 MiB page size exists and pdpe1gb is live (the CPU can use huge pages)
  C3 supply      HugePages_Free vs Total at runtime (fragmentation gauge, engine down or up)
  C4 THP mode    global THP policy (we are the hugeadm/DPDK class: reserved pool, THP stays madvise)
  C5 memlock     RLIMIT_MEMLOCK and the systemd user-service limit (mlock-based fallbacks need it)
  C6 RAM sanity  MemTotal vs the reservation cap (never size past MemTotal - max(16 GiB, 10%))

Sizing (the catalog formula, validated against the deployed box):
  demand = max over concurrency groups of sum(ceil(organ_bytes / 2 MiB))
  pool   = round up to whole GiB (512-page steps) of demand + 2 GiB slack
  cap    = (MemTotal - max(16 GiB, 10% MemTotal)) - already the binding general-RAM floor

Usage:
  python3 tools/strata-doctor.py                  # check + one-time explainers
  python3 tools/strata-doctor.py --list-checks
  python3 tools/strata-doctor.py --check C1 --verbose
  python3 tools/strata-doctor.py --json           # always full truth (no nag suppression)
  python3 tools/strata-doctor.py --fix            # print the paste-block for this device
  python3 tools/strata-doctor.py --fix --apply    # apply it (must run as root)

Exit codes: 0 armed+healthy, 1 degraded (pool short/THP off-class/memlock tight),
2 unarmed (no pool), 3 internal error.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

GiB = 1 << 30
HP = 2 << 20   # the 2 MiB page

# The organ catalog: per concurrency group (one engine serving at a time on this estate), the
# pool-drawing organs in bytes.  Sources: the engine's own measured boots (the DS `all` line,
# complement/arena boot lines, the store's memfd populate).  --catalog overrides.
DEFAULT_CATALOG = {
    "groups": [
        {"name": "gsq-iq3_s",     "organs": {"complement": int(37.97 * GiB), "store": int(31.64 * GiB), "kv": int(3.0 * GiB)}},
        {"name": "ud-iq2_xs",     "organs": {"arena": int(39.4 * GiB),  "store": int(31.64 * GiB)}},
        {"name": "ud-iq4_xs",     "organs": {"arena": int(56.3 * GiB),  "store": int(31.64 * GiB)}},
        {"name": "ds-engram",     "organs": {"copy_all_or_none": 110_850_000_000}, "all_or_none": True},
    ]
}

LEDGER = Path(os.environ.get("STRATA_DOCTOR_LEDGER", "/var/lib/strata/memory-ledger.json"))


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        # HugePages_* lines print bare counts; everything else prints "kB" (a kernel inconsistency
        # that once read this box's pool as 90152 GiB - measured, then fixed here)
        out[k.strip()] = int(parts[0]) * (1 if k.strip().startswith("HugePages") else 1024)
    return out


def cmdline_args() -> str:
    try:
        return Path("/proc/cmdline").read_text()
    except OSError:
        return ""


def thp_mode() -> str:
    try:
        m = re.search(r"\[(\w+)\]", Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text())
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def memlock_limit() -> int:
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return soft
    except (ImportError, OSError):
        return -1   # unknowable: report as unlimited-class rather than fail the check


def demand_pages(catalog: dict) -> tuple[int, str]:
    best, name = 0, ""
    for g in catalog["groups"]:
        pages = sum(-(-b // HP) for b in g["organs"].values())
        if pages > best:
            best, name = pages, g["name"]
    return best, name


def sized_pool(catalog: dict, mem_total: int) -> tuple[int, dict]:
    """demand + 2 GiB slack, rounded to whole GiB, capped by the general-RAM floor."""
    demand, who = demand_pages(catalog)
    slack = int(2 * GiB / HP)
    want = demand + slack
    want = -(-want // 512) * 512                      # whole-GiB steps (512 pages of 2 MiB)
    cap = (mem_total - max(16 * GiB, mem_total // 10)) // HP
    cap = -(-cap // 512) * 512
    capped = min(want, cap)
    return capped, {"demand_pages": demand, "binding_group": who, "want": want, "cap": cap, "capped": want > cap}


CHECKS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def run_checks(catalog: dict) -> list[dict]:
    mi = meminfo()
    total = mi.get("HugePages_Total", 0)
    free = mi.get("HugePages_Free", 0)
    want, math_ = sized_pool(catalog, mi["MemTotal"])
    size_kb = mi.get("Hugepagesize", 0) // 1024
    out = []

    out.append({"id": "C1", "name": "pool", "ok": total >= want, "state":
                "armed" if total >= want else ("partial" if total > 0 else "unarmed"),
                "detail": f"{total} pages armed ({total * HP / GiB:.0f} GiB) vs {want} required "
                          f"({want * HP / GiB:.0f} GiB; demand {math_['demand_pages']} pages from '{math_['binding_group']}' + 2 GiB slack)"
                + (f"  [CAPPED by the general-RAM floor from {math_['want']}]" if math_["capped"] else "")})

    out.append({"id": "C2", "name": "page-class", "ok": size_kb == 2048, "state":
                "ok" if size_kb == 2048 else "off",
                "detail": f"Hugepagesize {size_kb} KiB (2 MiB required; the engine's MAP_HUGETLB|MAP_HUGE_2MB draws depend on it)"})

    frag = (total - free) / total if total else 0
    out.append({"id": "C3", "name": "supply", "ok": total == 0 or frag < 0.95, "state":
                "ok" if total == 0 or frag < 0.95 else "exhausted",
                "detail": f"{free}/{total} pages free ({100 * frag:.0f}% drawn by running engines"
                          f"{'; all free with engines down' if total and free == total else ''})"})

    mode = thp_mode()
    out.append({"id": "C4", "name": "thp-mode", "ok": mode == "madvise", "state": mode,
                "detail": f"THP={mode} (this architecture is the reserved-pool class: THP stays 'madvise'; "
                          f"global 'always' is the DB-banned policy and not what backs our organs)"})

    ml = memlock_limit()
    out.append({"id": "C5", "name": "memlock", "ok": ml >= 64 * GiB or ml == -1, "state":
                "ok" if ml >= 64 * GiB or ml == -1 else "tight",
                "detail": f"RLIMIT_MEMLOCK {ml / GiB:.1f} GiB in this shell (large working-set locks want unlimited; "
                          f"check the user-service drop-in for the engine's session)"})

    out.append({"id": "C6", "name": "ram", "ok": math_["want"] <= math_["cap"] or not math_["capped"], "state":
                "ok" if not math_["capped"] else "capped",
                "detail": f"MemTotal {mi['MemTotal'] / GiB:.0f} GiB, reservation cap {math_['cap'] * HP / GiB:.0f} GiB "
                          f"(the general-RAM floor: OS + page cache + desktop + KV must live outside the pool)"})
    return out


EXPLAINERS = {
    "C1": (
        "The reserved 2 MiB page pool is {state} on this device.\n"
        "  The engine's streaming organs (expert arenas, the pinned complement, the shadow store)\n"
        "  draw guaranteed 2 MiB pages from it: pool memory cannot fragment, swap, or age, and engine\n"
        "  restarts re-arm perfectly - without it the hot pools silently fall to 4 KiB pages as the\n"
        "  box ages (10-25% lane decay, reboot-cured; the disease this architecture removes).\n"
        "  Fix for this device:\n{fix}"),
    "C4": (
        "Transparent huge pages are '{state}'.\n"
        "  This engine uses the RESERVED-pool class (PostgreSQL/Oracle/DPDK/vLLM pattern), not THP:\n"
        "  keep the global policy at 'madvise' - 'always' imports kernel stalls onto every mapping\n"
        "  on the box and is banned by the database class for exactly that.\n"
        "  Fix: sudo tee /sys/kernel/mm/transparent_hugepage/enabled <<< madvise"),
    "C5": (
        "Memory locking is tight ({state}).\n"
        "  The engine's working-set locks on large mappings charge RLIMIT_MEMLOCK; the modern driver's\n"
        "  GPU pinning does not, but the fallback paths do.  Unlimited is free and safe.\n"
        "  Fix: printf 'svg soft memlock unlimited\\nsvg hard memlock unlimited\\n' | sudo tee /etc/security/limits.d/90-strata.conf\n"
        "       sudo mkdir -p /usr/lib/systemd/system/user@.service.d\n"
        "       printf '[Service]\\nLimitMEMLOCK=infinity\\n' | sudo tee /usr/lib/systemd/system/user@.service.d/90-memlock.conf"),
}


def fix_block(pages: int) -> str:
    return ("--------------------------- cut here ---------------------------\n"
            f"sudo grubby --update-kernel=ALL --args=\"default_hugepagesz=2M hugepagesz=2M hugepages={pages} hugetlb_free_vmemmap=on\"\n"
            f"printf 'vm.nr_hugepages={pages}\\n' | sudo tee /etc/sysctl.d/91-strata-hugepages.conf\n"
            "printf 'svg soft memlock unlimited\\nsvg hard memlock unlimited\\n' | sudo tee /etc/security/limits.d/90-strata.conf\n"
            "sudo mkdir -p /usr/lib/systemd/system/user@.service.d\n"
            "printf '[Service]\\nLimitMEMLOCK=infinity\\n' | sudo tee /usr/lib/systemd/system/user@.service.d/90-memlock.conf\n"
            "------------------------- end cut here -------------------------\n"
            "Then reboot once. The pool re-arms itself at every future boot; engine restarts are the\n"
            "only recycle the engine ever needs.")


def verdict_hash(checks: list[dict]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps([{k: c[k] for k in ("id", "ok", "state")} for c in checks],
                                      sort_keys=True).encode()).hexdigest()[:16]


def nag_ledger(vh: str, checks: list[dict]) -> None:
    try:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    except OSError:
        return
    bad = [c for c in checks if not c["ok"]]
    if not bad:
        if prev.get("hash"):
            try:
                LEDGER.unlink()
            except OSError:
                pass
        return
    if prev.get("hash") == vh:
        return   # same bad state as last time: stay quiet (the nag law)
    try:
        LEDGER.write_text(json.dumps({"hash": vh, "ids": [c["id"] for c in bad]}))
    except OSError:
        pass


def main() -> int:
    args = sys.argv[1:]
    catalog = DEFAULT_CATALOG
    if "--catalog" in args:
        catalog = json.loads(Path(args[args.index("--catalog") + 1]).read_text())
    as_json = "--json" in args
    if "--list-checks" in args:
        print("\n".join(CHECKS))
        return 0
    only = args[args.index("--check") + 1] if "--check" in args else None

    checks = run_checks(catalog)
    if only:
        checks = [c for c in checks if c["id"] == only]
    mi = meminfo()
    want, _ = sized_pool(catalog, mi["MemTotal"])

    if "--fix" in args:
        print(fix_block(want))
        if "--apply" in args:
            if os.geteuid() != 0:
                print("--fix --apply must run as root; paste the block above instead", file=sys.stderr)
                return 3
            print("(apply path: run each line of the block; read-back verification is the block's own\n"
                  " next doctor run - DPDK's read-back-or-fail law)", file=sys.stderr)
        return 0

    if as_json:
        print(json.dumps({"checks": checks, "sized_pool_pages": want}, indent=1))
    else:
        for c in checks:
            mark = "OK " if c["ok"] else "!! "
            print(f"{mark}{c['id']} {c['name']:10s} {c['state']:8s} {c['detail']}")
        if "--verbose" not in args:
            nag_ledger(verdict_hash(checks), checks)
    worst = 0
    for c in checks:
        if not c["ok"]:
            worst = 2 if c["state"] in ("unarmed", "off") else 1 if worst < 2 else worst
    return worst


if __name__ == "__main__":
    sys.exit(main())
