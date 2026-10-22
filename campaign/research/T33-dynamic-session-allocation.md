# T33 — DYNAMIC SESSION ALLOCATION without added delay (design outline)

Written 2026-10-22 after F-271. Goal (user requirement): a 9-capable lane where the k-th slot
session's VRAM exists only while k clients are active — solo pays nothing, arrivals pay no delay.

## Why static fails (measured, F-271 clean re-tests)

`--batch 9` at boot: 9 × 0.73 GiB sessions + 9 × 51 MiB drafters are held FOREVER → pool
9,064 → 8,155 slots (−10%) AND the pl_ fast lane disengages for solo (its gate `pl_want = pipe &&
pl_pw >= 2` — pipeline windows and sessions compete for the same VRAM) → **solo −9.6%**
(182.4 vs 201.8, fresh profile both). Static allocation taxes the 99%-of-the-time case to serve
the 1%-of-the-time case.

## What already exists (all verified in code or measured tonight)

- `xcache.shrink(keep, err)` / `xcache.grow(bytes, err)` — runtime elasticity of the expert
  cache (generate.cpp ~11262/11272); the boot already self-resizes (F-242 era).
- Conversation migration between slots: **measured 34–59 ms** per conversation tonight
  ("slot 2 gave back 300 tokens ... in 35.7 ms") — hot-swap cost is tens of ms.
- Per-slot graph capture at boot ("captured the batch window over slots 8,8,0,0") — capture is
  cheap but must happen off the critical path.
- Admission machinery in server.py (ADMIT_SHORT_FIRST, ADMIT_MAX_WAIT_S) and the pl_ lane
  serving requests below 4 active slots (F-233's rule).

## The design — zero-added-delay contract

**Principle: arrivals are served INSTANTLY by existing capacity; session materialization is a
background promotion; demotion happens only on idle; resize never runs inside a window.**

1. **Pre-staged watermark (the headroom floor).** Cache auto-sizing keeps `k_empty × 0.78 GiB`
   unclaimed (start k=1). Today's sizing grows into every last byte — that is what starved the
   MTP draft head at reserve 600 (F-271) and what forces full pre-allocation of sessions. Cost of
   k=1 at the 97.7%-saturated hit term ≈ ~380 slots ≈ negligible M1 (the F-254 saturation finding).
   This one change also fixes the draft-head-starvation class of boot failures.
2. **Instant admission, lazy promotion.** Client #7..#9 arrives → served immediately on the pl_
   lane (requests below 4 active slots ride it anyway) or the admission queue; a background task
   then: shrink pool by one session's worth → allocate the session → capture its windows
   (~1–2 s, off the critical path) → migrate the conversation into the slot (34–59 ms measured).
   The arriving user generates from t0; peak-batching efficiency ramps within ~1–2 s. Cap
   promotions at 1 per ~2 s to avoid promote storms.
3. **Idle demotion with hysteresis.** Session idle > 2× the conversation hold → release buffers →
   `xcache.grow()` reclaims for the pool. Never demote mid-conversation; never demote while the
   admission queue is non-empty; require two consecutive idle checks (no flapping).
4. **The routing fix (required companion, independent of allocation).** The pl_ lane must stay
   preferred whenever active sessions < 4, regardless of the CONFIGURED parallel count — the F-233
   rule made explicit in the gate. Without this, dynamic allocation alone will not recover the
   −9.6% (tonight's static-9 penalty was pool-shrink + lane-disengagement together).
5. **Invariants (the no-delay contract, testable):**
   a. Admission NEVER waits on allocation (staged service starts at t0).
   b. shrink/grow never run with a window in flight (the existing resize fence).
   c. Arriving 7..#9 never invalidates graphs of slots 1..6.
   d. The draft head + engine floor stay respected (the watermark guarantees it).
6. **Gate + battery.** `STRATA_DYNAMIC_SESSIONS=1`, default off. Battery: standing anchor vs
   static-6 vs dynamic-9 — (i) solo five-boot means (must equal static-6), (ii) a staged 9-burst
   measuring time-to-full-throughput (expect ≤2 s ramp, zero refused/queued), (iii) idle-demotion
   flap test (promote→idle→demote→promote, no throughput dips), (iv) the F-271 hygiene law (fresh
   profile per battery; bursts re-teach the adapt tier — watch swap counts).
   Implementer-scale: ~1–2 days engine work (session lifecycle + watermark + promotion task).

## Alternatives considered and rejected

- Static 9 + accept −10% solo — violates the scoping law (AGENTS.md 4b) and the user requirement.
- Static 7–8 compromise — same class of tax, smaller.
- CUDA overcommit / lazy commit — the engine's pools are explicitly sized; no allocator hook.
- Two fixed partitions (solo pool + burst pool, no elasticity) — pays the slot tax permanently.

## Compositions

- Re-quant (+792 slots): widens the pool the watermark draws from — the two stack.
- 262k switch (−708 slots): dynamic sessions make even that switch's capacity elastic.
- ≥9-client conc ladder + COHORTS=3 re-test (F-249): this feature is the on-ramp for both.
