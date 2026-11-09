# VRAM deficit: the GSQ expert-cache slot drop (2026-11-09)

Verdict: the ~190-slot (-0.35 GiB) deficit is NOT an engine consumer and NOT the
vision elastic reserve. It is the device's free-VRAM baseline at boot: ~0.28-0.38 GiB
more VRAM held outside the engine since the Nov-8 reboot (desktop stack), and it
moves hour to hour with open apps.

## 1. Fresh boot capture (2026-11-09 11:36, GSQ lane 8081)

Captured from the operator's own in-flight `launch-custom-iq3s-8081.sh` boot
(battery `gsq-final.sh` leg r1); this audit booted nothing itself.
Full itemization: `campaign/results/vram-deficit-boot-2026-11-09.out` (from
`logs/custom-iq3_s.log`, boot at log line 679344).

- Banner: "filling the GPU's expert cache (**9900 experts, 18.78 GiB** of VRAM)"
- Sizing line: "expert cache auto: **20.87 GiB free**, 2000 MiB reserved
  (+141 MiB for the draft head) -> 7575 slots" (pre-elastic estimate)
- Identity: slots = (free_at_auto - 2000 MiB - 141 MiB) / 1.942 MiB per slot.
  20.87 - 1.953 - 0.138 = 18.78 GiB exact, every era.

VRAM consumers, fresh boot vs Nov-7 (`hostmem-leg0-iq3s-r1.out`, engine-log boot
~19:35 Nov-7, 9977 slots @ 18.93) vs Nov-7 leg2 evening (10044 @ 19.06) vs Nov-8
definitive (9845 @ 18.68): **every engine item byte-identical**:

| Consumer | Value (all eras) |
|---|---|
| dense weights (pack) | 1466 MiB |
| PLE projection matrices | 2018.88 MiB |
| 6 slot sessions (--batch 6) | 0.95 GiB each = 5.69 GiB |
| drafter (r17) weights | 874 MiB (experts 675, dense 111) |
| 6 slot drafters, private state | 51 MiB each |
| native Q5_K head | 521472000 B (497 MiB) |
| KV resident | 32768 of 262144 cells/layer in VRAM |
| verify graphs | 8 windows, ~6 MiB step (1453-1493 MiB free during capture) |
| free with everything loaded | 1259 MiB (Nov-8: 1272; gap = the deficit, post-fill) |
| vision (elastic) | 0 at fill time - see 2 |

## 2. Vision elastic reserve: EXONERATED

- Arming identical both eras: `--vision` in args (byte-identical "engine started:"
  lines Nov-7 20:20 through Nov-9 11:36); server narration "the expert cache keeps
  2048 MiB free for it" in every .out.
- The 2048 MiB is held ONLY while the encoder runs (`_vision_up`,
  serve/server.py:4364-4377 -> `engine.vram(reserve + elastic_mib)`), idle-stops
  after 120 s (`elastic_idle_s`). The boot-time fill subtracts only the standing
  2000 MiB + 141 MiB draft head - in BOTH eras. So it cannot explain a delta
  between eras.
- Knob (if repricing wanted): config key `"elastic_mib"` in the config's `vision`
  section (serve/server.py:2764, default 2048; no env var - only
  STRATA_VISION_READY_S/STRATA_VISION_ENCODE_S timing envs exist).
  One-line patch (NOT applied): add `"elastic_mib": 1024` to the `vision` section
  of `config/custom-iq3_s-8081.json`. Recovery: 1024 MiB / 1.942 MiB-slot =
  **~527 slots of image-time headroom** (the cache gives back ~527 fewer slots
  while the encoder runs). Boot-time slots recovered: **zero**.

## 3. Where the 0.35 GiB went

The deficit exists BEFORE the engine's first VRM reading: at the "--batch: 6 slot
sessions" checkpoint free VRAM was 19.90 GiB (Nov-7 20:29) -> 19.52 GiB (Nov-8
12:58) -> 19.61 (Nov-9 11:36); at the auto line 21.15 -> 20.77 -> 20.87. All
engine allocations above it are identical, so the gap is the cudaMemGetInfo
baseline - other GPU holders.

- Nov-8 reboot (hostmem hugepages arming) restarted the desktop stack; its VRAM
  grew. Measured today, engine down: **750 MiB total used** (Chrome GPU process
  98-153 MiB, ptyxis 36, nautilus 57, Wayland/compositor ~560 MiB).
- The baseline moves with apps: today's boots ranged free 20.58-20.91 GiB ->
  slots 9746-9915 (bisect legs lowest, quiet boot highest).
- Slot-era table (engine log, GSQ lane): Nov-6 (t22 era) 9120-9399 @ 17.30-17.83;
  Nov-7 leg0 9956-9977 @ 18.89-18.93; Nov-7 leg2 10019-10044 @ 19.01-19.06 (the
  high-water era); Nov-8 definitive 9836-9853 @ 18.66-18.69 (post-reboot drop,
  -0.37 GiB at auto); Nov-9 9746-9915 @ 18.49-18.81.
- Arithmetic: 0.35 GiB / 1.942 MiB-slot = **~184 slots** - the operator's -190.

Recovery options (evidence-ranked): close desktop GPU apps before banking boots
(free baseline returns, zero code); or `--vram-reserve-mib` below 2000 only if the
margin study re-approves it. The 10,044 era is reachable whenever the desktop is
quiet - it was never an engine regression.

## 4. MMLU quality gate, ready to run post-recovery (GSQ pack, port 8081)

Instrument: `campaign/mmlu-pro-gate.py` (70-question MMLU-Pro validation, draft
floor carried per request; writes `campaign/results/mmlu-<label>.json`).
Standing GSQ floor 0.933 (mmlu-1m-yarn-ml.json: 94.29, bar 93.3); 2x gate, plus
the 0.7 anchor for cross-era comparison (mmlu-anchor.json: 95.71).

```
cd /home/svg/LLM/strata-custom-2
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 vram-rec-r1
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 vram-rec-r2      # 2x gate: mean >= 93.3
.venv/bin/python campaign/mmlu-pro-gate.py 0.7   vram-rec-anchor  # vs 95.71 anchor
```
(STRATA_GATE_PORT defaults 8081 - the GSQ lane. No run executed here; prep only.)

## 5. Release state (named, not silently skipped)

This audit executed the cleanup pattern once (11:42) and killed one leg; operator
orchestrators respawned the lane within seconds: `bash campaign/batteries/
gsq-final.sh` pid 638304 (systemd --user, started 11:42:28, mid-battery r1-r5,
overwrites gsq-final-rN.out) and queued `campaign/batteries/ud-restore.sh`
pid 638771 (11:43:06). Engines now running belong to those batteries, not this
audit - this audit started no boot. Durable release requires stopping the
orchestrator pids first, then the standing pattern:
`pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s"; pkill -x strata`.
