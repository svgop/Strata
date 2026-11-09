---
type: research-digest
id: 2026-11-09-v5-memclock-governor
date: 2026-11-09
scope: strata-gpu-lock v5 design — SM keeps the v4 floor-range, MEM is hard-pinned at 14001 while an engine listens; line-level diff vs v4, falsifier dependency, install and rollback for the operator
subject: split-regime governor prototype (campaign/strata-gpu-lock-v5, new file only; v4 and the installed unit untouched)
method: design pass on top of the 2026-11-09-regression-clock-box audit; DRY=1 smoke tests of both MEM_PIN branches against a fake listener (no GPU state written, no engine booted); falsifier artifacts checked at /tmp
---

# v5 mem-clock governor: floor-range SM + hard-pinned MEM — 2026-11-09

## Verdict (read this first)

**v5 is built and smoke-tested; its deployment is gated on the same-boot falsifier, which is
still running.** Design: while an engine listens, SM stays v4's floor-range
(`-lgc 1800,3135` — hardware boost owns it, the v2 1780-plateau defect is not reused) and
MEM is HARD-PINNED (`-lmc 14001,14001` — min=max blocks mem DVFS, every burst enters at
full bandwidth). When no engine listens: full release (`-rgc -rmc`), unchanged. All v4
guardrails are byte-identical: watchdog dead-man (`systemd-notify WATCHDOG=1` each loop),
`-i 0` on every call, event-reason guardrail (`0x8|0x40|0x80` log-only; `0x400` normal),
thermal shed/resume (83/78 C), port-contract engine detection, GRACE=45 release,
SUPPORTED_CLOCKS ban. `MEM_PIN=1` is the script default; `MEM_PIN=0` in
`/etc/sysconfig/strata-gpu-lock` reverts the same script to exact v4 behavior.

## 1. Mechanism

- The deficit (+2 ms/window GPU-side, all lanes, ~-15%) appeared exactly at the v2→v3/v4
  cutover (Nov-07 15:23, journal-proven in [[2026-11-09-regression-clock-box]] §1.1).
- SM under v4 is NOT the suspect: burst entry is a microsecond DVFS step, sustained SM is
  ~2830-2850 cap-trimmed in both eras (audit §1.2, §4 — the Nov-07 digest grade-A proof).
- MEM is the one term v4 changed and the audit did not measure at burst granularity: under
  the range lock, sustained mem reads 13801 ("do not expect 14001 flat") and the Nov-09
  quiet-with-floor capture shows mclk floating 7001-13801. Hypothesis: memory-clock DVFS
  ramps slowly relative to decode bursts, so short GPU decode bursts under the range serve
  at whatever mem bin the card already holds, while v2's hard pin (`-lmc 14001,14001`)
  kept 14001 through every burst.
- Cost model: the pin blocks mem downclock too, so quiet-with-floor now holds mem at
  14001. GDDR static clock power while idle is small (watch dmon `p` after install —
  quiet-with-floor read 56-58 W under v4); switching power scales with traffic, hence
  "the pin costs watts only when data moves". The 575 W SW cap trims SM exactly as it did
  through the whole v2 era (Nov-7 dmon proof: pinned mem 14001 with SM sustained
  ~2830-2955, pviol=1) — sustained throughput has a working precedent on this exact box.
- Expected v5 sustained probe state = v2's sustained state (mem 14001, SM cap-trimmed)
  with v4's burst entry (no 1780 plateau). The governor emits `mem HARD PIN 14001` /
  `mem range 7001-14001` in every state-transition journal line so the eras stay
  separable in `journalctl -t strata-gpu-lock`.

## 2. Exact diff vs v4 (line-level, `diff -u campaign/strata-gpu-lock-v4 campaign/strata-gpu-lock-v5`)

Six functional hunks plus the rewritten header comment. Everything else is byte-identical.

1. **Header comment** (v4 lines 1-30 → v5 lines 1-34): narrative replaced; states the
   hypothesis, the split regime, and the MEM_PIN contract. No code.
2. **Defaults block** — one added line after `MEM_MAX=14001` (before the sysconfig source,
   so `/etc/sysconfig` overrides it):
   ```diff
    MEM_MAX=14001
   +MEM_PIN=1        # 1 = hard-pin mem at MEM_MAX while listening (v5); 0 = v4 range behavior
    TEMP_SHED=83
   ```
3. **New helper after `log()`**:
   ```diff
   +mem_mode() {   # journal-facing description of the active mem regime
   +    if [ "$MEM_PIN" = 1 ]; then printf 'HARD PIN %s' "$MEM_MAX"
   +    else printf 'range %s-%s' "$MEM_FLOOR" "$MEM_MAX"; fi
   +}
   ```
4. **`apply()` floor case** — the ONLY functional change. v4 line 81:
   ```diff
   -               "$NSMI" -i "$DEV" -lmc "$MEM_FLOOR,$MEM_MAX" >/dev/null 2>&1 || return 1 ;;
   +               if [ "$MEM_PIN" = 1 ]; then
   +                   # v5: hard pin min=max — mem DVFS cannot downshift/ramp inside a burst
   +                   "$NSMI" -i "$DEV" -lmc "$MEM_MAX,$MEM_MAX" >/dev/null 2>&1 || return 1
   +               else
   +                   # v4 behavior: range lock, DVFS owns the mem bin
   +                   "$NSMI" -i "$DEV" -lmc "$MEM_FLOOR,$MEM_MAX" >/dev/null 2>&1 || return 1
   +               fi ;;
   ```
   The pin target is `MEM_MAX`, which the existing remap (`MEM_MAX=${MEM_CLOCK:-$MEM_MAX}`)
   resolves to 14001 from the current sysconfig — operator-tunable, no new tunable path.
   `-pm 1` and the `-lgc` line above it are untouched. The `off` case is untouched.
5. **HUP trap** — one added line and the log extended (trap already re-applies floor, so a
   `MEM_PIN` flip lands on `systemctl reload` without a restart):
   ```diff
    SM_MAX=${SM_CLOCK:-3135}; MEM_MAX=${MEM_CLOCK:-14001}; SM_FLOOR=${SM_MID:-$SM_FLOOR}
   +MEM_PIN=${MEM_PIN:-1}
   -if [ "$state" != off ]; then apply floor && log "reload — floor re-applied (${SM_FLOOR}-${SM_MAX} sm, ${MEM_FLOOR}-${MEM_MAX} mem)"; fi' HUP
   +if [ "$state" != off ]; then apply floor && log "reload — floor re-applied (${SM_FLOOR}-${SM_MAX} sm, $(mem_mode) mem)"; fi' HUP
   ```
6. **Banner and three log lines** carry the mem regime (`watching v5 … mem $(mem_mode)`,
   `cooled to … `, `engine listening — clock floor … sm / mem $(mem_mode)`). The v4 banner
   fields (shed/resume/grace/poll/watchdog) are preserved verbatim.

Unchanged-by-construction: `engine_running` (port contract), `gpu_read` (batched
temp+event-reasons, no SUPPORTED_CLOCKS), `wd_ping`, TERM/INT release trap, the entire
state machine (shed, guard, gone counters), `sleep "$POLL"`, `systemd-notify --ready`.

Smoke evidence (this pass, DRY=1, fake listener on :18099, both branches):
`watching v5 (floor 1800-3135 sm / mem HARD PIN 14001 …)` → `engine listening — clock
floor 1800-3135 sm / mem HARD PIN 14001 …`; sed-copy `MEM_PIN=0` prints `mem range
7001-14001`; TERM trap logged `service stopped — clocks released`. `bash -n` clean on both.

## 3. Falsifier dependency

**FALSIFIER-PENDING** (2026-11-09 ~11:35 SAST): `/tmp/clk-floor.txt` and
`/tmp/clk-pin.txt` do not exist yet — the same-boot falsifier (floor-range probe vs
`-lmc 14001,14001` hard-pin probe) is still running. Decision rule, pre-registered in the
task: **pin leg ≥ +8% vs floor leg ⇒ hypothesis held** (mem-DVFS burst ramp is the
deficit); < +8% ⇒ not held, v5 stays shelved and the box-acquittal verdict of
[[2026-11-09-regression-clock-box]] (prewarm race + engine deltas) stands as the
explanation. Record both files' numbers here when they land, then act:
held → run the install block below; not held → leave v5 in campaign/ uninstalled.

## 4. Install block (operator)

The installed unit `/usr/lib/systemd/system/strata-gpu-lock.service` does NOT change —
its `ExecStartPre` (`-i 0 -rgc` / `-i 0 -rmc`, ignore-errors) and `ExecStopPost`
(identical pair) are exactly what v5 wants, and `Type=notify`/`WatchdogSec=30` are the
dead-man v5 keeps pinging. Only the script swaps:

```bash
sudo cp /home/svg/LLM/strata-custom-2/campaign/strata-gpu-lock-v5 /usr/libexec/strata-gpu-lock
sudo systemctl restart strata-gpu-lock          # daemon-reload not needed (no unit change)
journalctl -t strata-gpu-lock -n 2 --no-pager   # expect: watching v5 (… mem HARD PIN 14001 …)
```

Optional pre-verify without touching the installed service:
`bash -n campaign/strata-gpu-lock-v5` (done this pass) and a DRY smoke (done this pass).

## 5. Rollback

- Soft (no file moves): `echo 'MEM_PIN=0' | sudo tee -a /etc/sysconfig/strata-gpu-lock`
  then `sudo systemctl reload strata-gpu-lock` — the HUP trap re-sources, re-applies the
  floor in v4 mem-range mode, and logs `mem range 7001-14001`. Remove the line + reload
  to go back to pin. Caveat: the trap keeps the last-sourced `MEM_PIN` if the line is
  later deleted from the file; a restart re-anchors the default (`1`).
- Hard (exact v4 restoration): `sudo cp /home/svg/LLM/strata-custom-2/campaign/strata-gpu-lock-v4 /usr/libexec/strata-gpu-lock && sudo systemctl restart strata-gpu-lock`
  — campaign v4 is byte-identical to the currently installed script (verified this pass),
  so this returns the box to the Nov-08 state with zero drift.

## Provenance

- Read: `campaign/strata-gpu-lock-v4` + `.service` (installed copy verified byte-identical
  via diff against `/usr/libexec/strata-gpu-lock`), `/etc/sysconfig/strata-gpu-lock`
  (v2 tunables: `MEM_CLOCK=14001` → pin target),
  `/home/svg/LLM/strata-custom-2/.docs/research/digest/2026-11-09-regression-clock-box.md`
  (era timeline, mem 13801-sustained evidence, quiet-with-floor mclk 7001-13801 capture),
  `systemctl show strata-gpu-lock` (ExecStartPre/StopPost argv).
- Wrote: `campaign/strata-gpu-lock-v5` (new file; v4 files and the installed unit/service
  untouched), this digest.
- Smoke this pass: `bash -n` both branches; DRY=1 runs with a throwaway :18099 listener;
  journal reads of TAG `v5-smoke-pin` / `v5-smoke-range`. No GPU state written; falsifier
  artifacts (`/tmp/clk-floor.txt`, `/tmp/clk-pin.txt`) checked and absent.
