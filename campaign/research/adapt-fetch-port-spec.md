# ADAPT_FETCH port spec — the predicted-prefetch arm, priced before it runs (2026-11-20)

Status: SPEC, precondition-gated (the STRATA_DM_WAIT_STATS recalibration boot prices it first; law 15 —
an arm whose priced expectation is inside the drift band is not run). Source: upstream v0.2.16
(`/home/svg/LLM/strata`), sites read 2026-11-20. Queue position: after H3 verdict, H4, the wait-stats
read (the succession order, SVG-CUSTOM note 14:12).

## 1. The mechanism (upstream's, read at the sites)

The PCIe miss-share becomes a PREDICTED PREFETCH with tier admission, in one pass:

1. **Rank**: the pool's plan builder sorts the layer's misses (not tail-skipped, not host-resident,
   PINNED blobs only) by the tier's USAGE counts — `std::stable_sort` on `use[ids[...]]`, descending
   (`expert_source.cpp:5072-5080`). The usage table is the predictor; no new model is introduced.
2. **Pick**: the top `m` (the PCIe share the balancer already computed) are marked for fetch.
3. **Admit**: as their bytes are staged by `fetch_blobs`, the destination IS a freed tier slot —
   the plan's new `dst2` column (`verify.cpp:1051-1064`, `2 * kStagingBlobs` u64s at
   `dst2_off_`) carries per-entry slot addresses; `rebase_ptrs(..., p_dst2)` points the execution
   at the slot. The fetch is the admission: no second copy, no extra pass.
4. **Gate**: `STRATA_ADAPT_FETCH` env — 0 = off (upstream default; ours would ship the same),
   1 = admission beside the tier's own copies, 2 = instead of them (counts still decay).
   `STRATA_ADAPT_FETCH_MIN` / `_MARGIN` (default 1 / 0.5) are the admission thresholds.
   Hard dependency: `plan->pcie_mode == 2` (`expert_source.cpp:4781`) — the staged-copy PCIe mode.

## 2. The gap map (our fork)

Our tree already owns every neighbor: `STRATA_PCIE_MIN1` (ported, `expert_source.cpp:4845`),
`STRATA_ADAPT_CHECKRES` (ported, `generate.cpp:1651`), the balancer (`m_bal_` stamped waits), and
the `pcie_mode == 2` staging machinery (`verify.cpp:2276-2281`). The port is exactly three sites:

| Site | Change | Upstream anchor |
|---|---|---|
| `src/core/verify.cpp` plan geometry | add `dst2_off_` column + `plan_i32_` growth + `sink_.dst2` (two init sites: ~1051, ~3742-equivalent) | 1051-1064, 3742 |
| `src/core/verify.cpp` GPU submit | thread `p_dst2` through `fetch_blobs` / `rebase_ptrs` (the pcie_mode==2 block we already have) | 2426-2432 |
| `src/core/expert_source.cpp` pool plan | `fetch_can()` gate + the usage-ranked pick + `P.dst2` fill (`fetch_admit` member on the dispatch struct) | 4781, 5059-5080, 5141 |
| `src/program/generate.cpp` env | `adapt_fetch()` accessor (0/1/2) + MIN/MARGIN + the `fetch_admit` wiring into the dispatch | 2236-2239 |

## 3. The port order (Mikado; each step compiles, links, boots)

1. Plan geometry + `sink_.dst2` plumbed but ALWAYS EMPTY (dst2 null) — no behavior change;
   bit-check green. The plan is a load-bearing layout: this step only WIDENS it.
   **[LANDED 2026-11-20, commit 72a3fc058 — plan carries the dst2 column at both init sites,
   GpuPlanSink owns the member; builds clean ecc0a3b5; proof owed with step 2 in one boot.]**
2. `p_dst2` threading with a null default — `fetch_blobs`/`rebase_ptrs` signature grows a
   defaulted arg; the staged path behavior identical.
   **[LANDED 2026-11-20, commit 8adb0a5aa — kernels carry the per-blob destination select
   (dst2[k] != 0 ? the tier slot : staging, kd-cached per blob), wrappers + declarations
   null-defaulted; builds clean b8153a66; the one boot proves 1+2 together.]**
3. Pool-side rank/pick behind `STRATA_ADAPT_FETCH=0` (default off). **[LANDED 2026-11-20,
   commits e4c76f4b6 (3a: fields/env) + 64a79c18b (3b: admission + the faithful arm gate).]**
   **Applicability fact discovered in porting: the arm REFUSES the resident-RAM mode and the DMA
   PCIe mode (upstream's own gate, the reason named in the banner) - our standing runs BOTH, so
   the battery needs a scoped staged-copy recipe first (law 4b). The env-off proof boot is the
   only remaining gate before deploy.** `ADAPT_CHECKRES=1` boot verifies the residency table stays exact with the arm off.
   **The complete anatomy (read at upstream 5125-5205, executes mechanically):**
   - Dispatch fields to port: `fetch_admit` (int), `fetch_min`/`fetch_margin` (floats),
     `fetch_avail` (per-layer flags), `fetch_mark` (per-expert scratch), `in_flight`
     (per-layer-expert guard), `res_mut` (the MUTABLE residency table - distinct from host_res
     if our fork's is read-only), `cache_base`/`cache_slot_off`/`cache_blob` (the VRAM cache
     geometry), `fetch_admitted` (counter), `res_dirty`/`res_dirty_idx`, `fetch_log*`.
   - Default `P.dst2[q] = 0` (staging) for all fetches when `fetch_admit > 0 && P.dst2`.
   - The pick: candidates `cq` = fetched misses with `use[e] >= fetch_min` and not in-flight,
     stable-sorted by usage desc; victims `vic` = resident experts the window does NOT route
     (`res_mut >= 0`, `fetch_mark == 0`), kept least-routed-first via the insertion-swap.
   - The admission (margin-gated swap): `use[e] >= use[v] + fetch_margin`, `P.dst2[q] =
     cache_base + slot_off[slot]`, `res_mut[v] = -1; res_mut[e] = slot`, dirty-idx appended,
     `in_flight` set for the layer-expert until the fetch lands.
   - Our miss-pick rule today is POSITIONAL (the last m in routing order, `miss_rank >= nmiss-m`);
     upstream's `fetch_on` pick is usage-ranked - compose per upstream (the rank picks, the
     positional rule stays for the non-admitted tail).
4. Arm on (=1): the parity harness + the battery (below).

## 4. Verification (the harness before the arm)

- **Parity**: admission changes residency, so outputs are float-rounding-identical AT BEST
  (CPU vs GPU kernel order, the s2 header's own contract) — bit-check is NOT the oracle. The
  oracle is `ADAPT_CHECKRES` (the residency-table read-back must report ZERO mismatches — an
  executable invariant on the admission itself) + M3b in-band + the MMLU gate.
- **Battery**: `config-ladder`/`env-ladder` 3v3, `STRATA_ADAPT_FETCH=1` vs 0, load per leg,
  all lanes + M3b. The mechanism metric: the window anatomy's `cpu-exp` share (the 2.73-2.90 ms
  term) and the dm-wait-stats `waitCPU` column must FALL by the priced amount.

## 5. The pricing formula (fill from the wait-stats boot)

Priced effect = (miss entries converted to hits per window) × (per-entry CPU miss cost) −
(fetch/admission overhead). Inputs wanted from the recalibration boot: waitA / waitB / waitCPU
per group, reach-before-wait for each, and the per-layer waitCPU spread. If waitCPU's exposed
(the un-hidden) share is X ms/window and the top-m usage-ranked misses cover fraction f of it,
the ceiling is X·f on M1's window — the arm runs only if that ceiling clears the +4% bar at a
plausible f (the usage table's hit share ~0.90 at champion calibration says f can be large).
Kill conditions: pinned-blob coverage too small (the picker requires PINNED blobs), pcie share
m collapses to 0 at typical decode (PCIE_MIN1 exists for exactly this), or admission thrash
(MIN/MARGIN cannot hold a slot — visible as fetch-count >> distinct-fetched).

## 6. Cost and rollback

Three files, ~150 lines lifted from a production tree we have already ported two siblings from
(PCIE_MIN1, CHECKRES). Rollback: `STRATA_ADAPT_FETCH=0` (the shipped default) — one env, no
binary swap needed. The plan-widening step (1) is the only structural footprint and is inert.
