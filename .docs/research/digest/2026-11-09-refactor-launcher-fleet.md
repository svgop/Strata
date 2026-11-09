---
type: digest
id: 2026-11-09-refactor-launcher-fleet
date: 2026-11-09
scope: launch-custom-{iq3s,ud-iq2xs,ud-iq4xs,dsv41,dsv41-4e}-8081.sh + campaign/shm-shadow.sh
method: full read of all 5 launchers + shm-shadow.sh; pairwise diff; git archaeology (a01efdca, 901f8409, ea56db09, 91a1e5da); consumer grep (campaign batteries, tools/strata-doctor.py)
status: study complete — consolidation ready, adoption gated on validation protocol
---

# Launcher fleet duplication study

Five launchers serve port 8081 (swap-by-stop port law): three Qwen quant lanes (`launch-custom-iq3s-8081.sh` 132 L, `launch-custom-ud-iq2xs-8081.sh` 127 L, `launch-custom-ud-iq4xs-8081.sh` 127 L) and two DeepSeek-V4.1 lanes (`launch-custom-dsv41-8081.sh`, `launch-custom-dsv41-4e-8081.sh`, 29 L each). All six files pass `bash -n` today (verified 2026-11-09).

## 1. Drift matrix

Blocks audited: swap-guard scope, taskset pin, single-owner store rule, shadow re-establish call, env battery, prewarm/health waits. "Identical" = byte-identical modulo the per-launcher config path.

| Block | iq3s | iq2xs | iq4xs | dsv41 | dsv41-4e | Verdict |
|---|---|---|---|---|---|---|
| B1 swap-guard condition line | :129 | :121 | :121 | :23 | :23 | **IDENTICAL** — `${STRATA_SWAP_GUARD-1}` + `command -v systemd-run` + `${XDG_RUNTIME_DIR:-}`, unit `strata-engine-$$`, `-p MemorySwapMax=0` in all 5 |
| B1b swap-guard position | after pin (:129) | **before pin** (:121) | **before pin** (:121) | **before pin** (:23) | **before pin** (:23) | **DRIFTED — behavior-changing** (see D1) |
| B2 taskset gate form | multi-line `if` + 5-line CCD comment (:117-124) | one-line `if` (:124-126) | one-line `if` (:124-126) | one-line `if` (:26-28) | one-line `if` (:26-28) | **DRIFTED — cosmetic** (see D2) |
| B3 single-owner rule (`HugePages_Free >= 17000`) | :67-78 | :67-78 | :67-78 | absent | absent | **IDENTICAL across trio**; dsv41 absence is correct (no shadow store on DS lanes; dsv41 has 0 `HugePages` refs) |
| B4 shadow re-establish call | :114 | :114 | :114 | absent | absent | **IDENTICAL across trio, including a shared typo** (see D4) |
| B5 env battery | :13-109 | :13-109 | :13-109 | :15-16 only | :15-16 only | **IDENTICAL across trio** (diff-proven); dsv41 pair carries its own 2 exports, differs only in config name |
| B6 prewarm/health waits | absent | absent | absent | absent | absent | **PREMISE STALE** — zero `prewarm`/`health`/`curl`/`sleep` hits in all 5; that block lives in the battery harnesses (`campaign/batteries/gsq-pinning-ab.sh:14` curl /health loop), not the launchers |
| B7 header comment | "(Qwen3.8 IQ3_S…)" :2 | **stale IQ3_S header** :2 | **stale IQ3_S header** :2 | correct per-file :2 | correct per-file :2 | **DRIFTED** (see D3) |

### D1 — the behavior-changing drift: guard-vs-pin priority is split across the fleet (SAFE-NOW to observe, the fix is NEEDS-BENCH)

Every exec branch terminates the script, so the swap guard and the taskset pin are mutually exclusive in all 5 launchers — but **which one wins differs**:

- iq3s pins **first** (`launch-custom-iq3s-8081.sh:121-124`), so `STRATA_ENGINE_CPUS=8-15,24-31` on this systemd host execs taskset and **never evaluates the guard** — the engine boots swap-eligible (no `MemorySwapMax=0`). The comment at :120 ("STRATA_ENGINE_CPUS overrides") documents this branch.
- iq2xs/iq4xs/dsv41/dsv41-4e guard **first** (`launch-custom-ud-iq2xs-8081.sh:121-123` etc.), so the same env on a systemd host is **silently ignored** — no taskset ever appears in the cmdline, contradicting the inline promise "set 8-15,24-31 for contended hours" (`launch-custom-ud-iq2xs-8081.sh:124`).
- Neither design composes the two; there is no launcher where pin + `MemorySwapMax=0` coexist.

Origin is insertion order, not decision: the pin was born on iq3s alone (a01efdca, 2026-11-09 19:13), the guard landed on all 5 later the same day (ea56db09) and was **appended after** the pin in iq3s; the pin was then spread to the other 4 (91a1e5da, "taskset pin wrapper made consistent across all five launchers") **after** their guards. "Made consistent" fixed the args, not the structure.

### D2 — cosmetic drift: iq3s carries the multi-line pin form + CCD comment the siblings lack

iq3s:117-124 = 5-line CCD topology comment + multi-line `if [ -n "$_CPUS" ]; then\n  exec taskset…\nfi`. The four siblings carry only the one-line form and the short inline comment (`launch-custom-ud-iq2xs-8081.sh:124`). Semantically identical forms; structurally the fleet already disagrees 24 h after the block was born.

### D3 — stale headers

`launch-custom-ud-iq2xs-8081.sh:2` and `launch-custom-ud-iq4xs-8081.sh:2` both still read "savage-custom v2 launcher, Linux port (Qwen3.8 IQ3_S + r17 modded draft, port 8081)" — copy-paste from iq3s never updated. Comment drift is trust drift.

### D4 — shared awk typo in the shadow gate (trio only, identical in all 3)

All three quant launchers gate the shadow re-establish with a **double-brace** awk program: `awk '/^HugePages_Free/{{print $2}}'` (`launch-custom-iq3s-8081.sh:114`, iq2xs:114, iq4xs:114). Valid awk (nested action block, same output) but the sibling single-owner check 44 lines up uses the correct `{print $2}` (:70) — proof of copy-paste editing with local fixes. Harmless today; a hazard the lib kills.

### D5 — cwd-dependent shadow re-establish (latent, trio only)

iq3s:114 runs `bash campaign/shm-shadow.sh` **relative to the invoking cwd**, and the `cd "$(dirname "$0")"` lands only at :116, after it. Invoked from outside the repo root (all battery harnesses invoke from root: `campaign/env-ladder.sh:31`, `campaign/binary-ladder.sh:23`), the copy silently skips (`|| true`) and the boot proceeds on the disk-path fallback. dsv41 cds first (`launch-custom-dsv41-8081.sh:18`) and has no relative call, so it is clean.

### Git archaeology — the malformed-taskset event

- **a01efdca** (2026-11-09 19:13), iq3s only: pin born, default **ON** (`_CPUS="${STRATA_ENGINE_CPUS-8-15,24-31}"`), multi-line form, no guard yet. The commit message says "syntax repaired" — the malformed branch lived in the working tree and was fixed **pre-commit**; the broken text itself is not recoverable from git, the message is the only witness. This is the "one launcher had a malformed taskset branch its siblings lacked" event: it happened on iq3s, and the siblings had nothing to be malformed *from* yet.
- **901f8409**: default flipped ON→OFF ("taskset pinning neutral on a quiet box = parked env-gated default-off").
- **ea56db09**: guard added to all 5 ("verified live: engine under strata-engine-*.scope with swap.max=0") — appended after iq3s's pin, before the other 4's future pins.
- **91a1e5da**: pin spread to all 5, "env-gated, correct args per launcher" — the spread reproduced the semantic gate but not iq3s's structure (D2) nor a common ordering (D1).

Four edits in one day across five copies of the same 8-line block; one syntax incident, one ordering split, two stale headers. The fleet already demonstrated the failure mode the refactor removes.

## 2. Consolidation plan — `launch-common.sh` (sourced lib)

New file `launch-common.sh` at repo root, sourced by each launcher after its header. Per-launcher shrink: trio ~35 lines → ~10; dsv41 ~12 → ~8. The exec ladder exists exactly once.

### Moves INTO the lib (owned once)

1. **`strata_launch_init`** — `cd "$(dirname "$0")"` at lib top. Kills D5 (cwd-dependent shadow call) as a side effect.
2. **`strata_shadow_store_select`** — the single-owner ladder verbatim from iq3s:65-82 (`_SHADOW_CANON`, `_SHADOW_TMPFS`, `HugePages_Free >= 17000` → canon, elif tmpfs, elif canon, else empty; the `-` not `:-` export note at :79-81 preserved verbatim — an explicit empty `STRATA_SHADOW_EXPERTS=""` must survive as rollback). Called only when the launcher sets `STRATA_LAUNCH_SHADOW=1`. The `[ -n … ] && echo` announce line (:82) stays inside it.
3. **`strata_shadow_reestablish`** — the iq3s:114 gate, with the `{{print $2}}` → `{print $2}` fix (output-identical) and now cwd-safe.
4. **`strata_exec_engine CONFIG`** — the exec ladder, priority **fixed fleet-wide**: guard first; when `STRATA_ENGINE_CPUS` is also non-empty, **compose**: `exec systemd-run --user --scope --collect -q -p MemorySwapMax=0 --unit="strata-engine-$$" taskset -c "$_CPUS" .venv/bin/python serve/server.py --engine strata --config "$1" --port 8081 --host 0.0.0.0`. Guard-only, pin-only (`STRATA_SWAP_GUARD=0`), and plain-exec fallback branches verbatim from the current condition lines. Composing is the one deliberate behavior change: it fixes D1 in the direction both comments promise (pin works under systemd; guard never silently dropped). Capability strictly grows (law 16); boot-path cost unchanged (law 13).
5. A lib-only `STRATA_LAUNCH_DRY=1` branch that prints the composed argv instead of exec — the validation harness (§3). Default unset, invisible in production.

### Stays PER-LAUNCHER

1. Shebang + the header doc-comments (each launcher's measured-arms log is per-lane evidence; iq3s:5-109 must not be hoisted — it is the lane's recipe, and hoisting would re-couple quant recipes the fleet deliberately keeps independent).
2. The env battery exports (trio identical today :13-109, but per-lane by design; dsv41's `DS4_ATTN_SPLIT=1` + `STRATA_DS4_SLOT_DIR` :15-16).
3. Three variables and the final call:
   ```bash
   STRATA_LAUNCH_CONFIG=config/custom-iq3_s-8081.json   # per launcher
   STRATA_LAUNCH_SHADOW=1                               # trio only; unset on dsv41 pair
   . "$(dirname "$0")/launch-common.sh"
   strata_exec_engine "$STRATA_LAUNCH_CONFIG"
   ```
4. Header fix while touching: iq2xs:2 / iq4xs:2 get their real quant names (D3).

### Preserved invariants

- All env gates byte-preserved: `${STRATA_SWAP_GUARD-1}`, `${STRATA_ENGINE_CPUS-}`, `${XDG_RUNTIME_DIR:-}`, `${STRATA_SHADOW_EXPERTS-$_shadow}` (`-`, not `:-`), `STRATA_COHORTS`/`STRATA_POOL_SPIN_US`/… passthrough.
- Consumers unchanged: batteries invoke by path with env prefixes (`campaign/env-ladder.sh:31-35`, `campaign/batteries/t64-battery.sh:14`, `campaign/binary-ladder.sh:23`) — none source the launchers, none pass positional args; the lib is invisible to them.
- `tools/strata-doctor.py` C7 reads the **running process's cgroup** (`tools/strata-doctor.py:159-185`), launcher-agnostic — unaffected, and it becomes the live witness.
- Plain-exec fallback retained (non-systemd hosts, `STRATA_SWAP_GUARD=0`).
- `campaign/shm-shadow.sh` itself untouched (its own fast-path/verify logic is single-copy already).

## 3. Validation protocol (all must pass before adoption stands)

Static (SAFE-NOW, no boot needed):

1. `bash -n` on all 5 launchers + `launch-common.sh` (gate already green pre-refactor — keep it green).
2. `shellcheck` on the lib + 5 launchers — new gate; today's fleet has never had one (D4's typo survived 5 days because of that).
3. **Argv equivalence matrix**: `STRATA_LAUNCH_DRY=1` runs, 4 env combos × 5 launchers (unset/unset, `STRATA_ENGINE_CPUS=8-15,24-31`, `STRATA_SWAP_GUARD=0`, both) — the printed argv must match the pre-refactor exec line for engine/config/port/host byte-for-byte; only the wrapper prefix may differ, and only per the compose rule.

Live (NEEDS-BENCH — one boot per family, three boots total):

4. **Guard witness**: boot iq3s (quant family) and dsv41 (DS family) → engine pid under `strata-engine-*.scope` (`systemd-cgls` / `/sys/fs/cgroup/user.slice/…/strata-engine-*.scope` present) and `memory.swap.max` reads `0`; `tools/strata-doctor.py` C7 green.
5. **Pin witness**: same boot with `STRATA_ENGINE_CPUS=8-15,24-31` → `cat /proc/<pid>/status | grep Cpus_allowed_list` == `8-15,24-31` **and** the scope witness still green (this is the D1 fix proving composition) — on **both** an iq3s boot (order flips here) and an iq2xs/iq4xs boot (pin starts working here).
6. **Fallback witness**: `STRATA_SWAP_GUARD=0` boot → no scope, server up on 8081.
7. **Shadow witness** (trio boot): launcher stderr line `launcher: shadow store …` names the expected tier (canon under `HugePages_Free >= 17000`, `/dev/shm` tier otherwise) — same tier selection as the pre-refactor boot.
8. **Law-13 lane check**: same-hour ABAB, one `campaign/probe-lane.py --short-n 6` leg pre/post on the GSQ lane. Expected: inside noise (runtime deltas are cd-order and the awk brace fix, both inert). Any leg worse → revert the commit; the refactor is one commit, `git revert` restores all 5 launchers.

## 4. Verdict

**Worth it.** Conditions: land as one commit; the D1 compose fix boots per §3.4-5 before it stands; strict byte-preservation of every env gate.

- **Maintenance saved**: the fleet already burned four same-day edits and one syntax incident on the 8-line pin block (a01efdca→91a1e5da, all 2026-11-09); the guard block is 5 copies of 4 lines; the store ladder is 3 copies of 18 lines. Every future guard/pin/store change lands once, with one validation matrix instead of five.
- **Risk added**: low-moderate and bounded — the exec ladder is the only subtle machinery, it consolidates into one reviewed function, and the DRY argv matrix + doctor C7 give byte-level and kernel-level witnesses. The single behavior change (D1) is a bug fix both code-paths' comments already claim.
- **Not included**: hoisting the env battery (per-lane by design), touching `campaign/shm-shadow.sh`, or adopting health-wait logic into launchers (it belongs to the battery harnesses — B6 premise stays stale).

Findings index: D1 behavior split (fix NEEDS-BENCH) · D2 cosmetic drift (SAFE-NOW) · D3 stale headers (SAFE-NOW) · D4 awk typo (SAFE-NOW) · D5 cwd-dependent shadow call (SAFE-NOW to fix) · B6 prewarm/health premise stale · a01efdca malformed-branch event (historical, message-only evidence).
