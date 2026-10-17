# T15C — OVERLOOKED WINS: the never-swept env inventory, re-priced candidates the dossiers missed, and stale defaults

Date: 2026-10-16/17. Scope: READ-ONLY hunt for 5-10%-class wins NOT already banked by F-200..F-233 or
T13A/B/C + T14A/B/C (everything re-found there is marked **known** and dropped from the ranking).
Calibration: the F-233 standing recipe (launch-custom-iq3s-8081.sh + custom-iq3_s-8081.json, five-boot
means M1 198.5 / pair 197.6 / conc 312.4, acceptance 74.9, MMLU-Pro 95.0). Hard laws honored: no
prefill-bits changes priced without the F-230 tax; no accounting-identity estimates (F-232) — every
candidate below carries a file:line mechanism read or is explicitly marked inference; marginal verify
row = 3.56 ms M1 / 4.62 conc (T14C); node-count cuts alone are inert (F-205/F-206); the CPU pool is
hidden only by the cohort escape (F-223/F-233). No server was touched; nothing was built or booted.

---

## 1. Summary

The dossiers mined the big lines (hc reads, VRAM hits, attention, doorbell tail, cohorts) but left
three seams essentially untouched:

1. **The S23/S25/S26 decode-kernel package is in the tree, live on CUDA, shipped BY DEFAULT on AMD
   gfx1151/gfx12 — and has never been armed once on this 5090** (`STRATA_Q6_PACKED`, `STRATA_TSUM`,
   `STRATA_HC_Q8`; zero hits in every launcher, boot log, and result file). Cross-arch precedents are
   in-file: +2.3% (S26 Q6/Q8 packed, Strix Halo), +2.5-4% decode (HC_Q8 on RDNA4, native_dense.cpp
   comment). These attack the current TOP kernel line (hc reads 2.3-5.2 ms/window, F-218) and the
   head/dense GEMV lines, and two of the three are bit-identical class.
2. **A whole class of arms is officially UNTESTED on the true base** because F-205 voided every
   bare-based verdict and only some were re-run: L2_SETASIDE=72 (explicitly), the DT peak location
   (explicitly "optional re-sweep", never done), and the pcie-frac 0.35 constant (tuned in a ~30-miss/
   layer era; at 97.7%+ hits the share rounds to 0 on most miss-layers and the engine's own PcieAuto
   tuner — built for exactly this — sits unengaged because the config passes an explicit frac).
3. **The conc launcher carries a stale interaction**: `STRATA_ATTN_LANECELL=1` (F-215) has never been
   separated from ATTN_DAILY (F-221 changed the same kernel family, heavy attention 4.09→1.96 ms) or
   from COHORTS (F-233's control was the LANECELL launcher itself). Its +4.6% conc premise is
   pre-DAILY evidence about a kernel that no longer exists.

Plus one unexecuted bit-identical micro (T13A #8's second sampler sync — still in the code, never
banked), one designed-but-never-built kernel item that cuts c_row (the GDN decode step ILP rewrite,
F-201 R27 — still open, still latency-bound by its own header comment), and a handful of stale
comments/defaults (§4). Nothing found rises to +10% alone; the realistic aggregate of the never-swept
class is +2-6% across lanes for ~6 boots of env-only work — the cheapest tranche left on the board.

---

## 2. (a) The env-knob inventory

417 distinct `STRATA_*` knobs at 549 `getenv` sites (grep of src/ + include/, this tree @ F-233).
25 are STANDING (the launcher). Of the rest, ~30 have sweep/verdict history in the journal; the rest
have NEVER been set on this rig in any banked boot. The full list with per-knob status tags is the
appendix at the end of this file (name @ site, [tag] where status is known). This section carries the
perf-relevant shortlist with defaults read from code; "never-swept" means no banked boot on THIS rig
at THIS recipe.

### 2.1 Shortlist — perf-relevant, never-swept (the candidates' home)

| knob | site (read this session) | default | what it does | status |
|---|---|---|---|---|
| `STRATA_Q6_PACKED` | native_mmvq.cu:2027, pack at :2004-2040, dispatch :2392; head wired native_head.cpp:75 | off (CUDA); **default ON for gfx1151** (arch_defaults.cpp:47, "packed Q6_K heads (S26)") | load-time plane-repack of Q6_K heads (ql\|qh\|scales\|d); packed MMVQ kernel intercepts `native_q6_k_mmvq` | never armed here |
| `STRATA_Q6P_SELFTEST` | native_mmvq.cu:2064-2090 | off | in-boot bitwise memcmp vs GGUF-layout kernels (nc 1..4) + **paired timings** on the real shapes | the free falsifier for the row above |
| `STRATA_TSUM` | native_mmvq.cu:1158-1162, used :1784, :1917 | off (CUDA); **default ON for gfx1151** (arch_defaults.cpp:51) | multi-column MMVQ warp-sums as one transposed butterfly; **bitwise** | never armed here |
| `STRATA_Q8_PACKED` | native_mmvq.cu:1992, native_dense.cpp:160 | off (CUDA); ON for gfx1151 (+2.3% S26) | packed Q8_0 decode layout | inert on this pack: dense tensors are Q6_K (T13C artifact parse) — no eligible Q8_0 shapes |
| `STRATA_HC_Q8` | native_dense.cpp:45-140, hc_q8_name :71-78 | off (CUDA); **auto-ON for gfx12/RDNA4**: "decode +2.5 to +4%, KL 0.0003-0.0010" | hc projections (hc_attn/ffn up/down/inject + final mixer) read as Q8_0 (bf16→Q8_0 at load) — halves the bytes of the hc read line | never armed here |
| `STRATA_PCIE_MIN1` | expert_source.cpp:3871-3875 | off | with ≥2 misses/layer, give the PCIe lane ≥1 blob even when the 0.35 share rounds to 0 | never swept; newly relevant at 97.7%+ hits (§3 #6) |
| `STRATA_MMVQ_IL_ROWS` / `_T1` | native_mmvq.cu:2603, :2643 | runtime IL tables (sm_120 table in docs/MMVQ_IL_TABLE.md) | per-shape IL row-count override; T=1 rows class | never swept on sm_120; the T13A #6/F-232 microbench question |
| `STRATA_POOL_SPIN_US` | pool.cpp:478 (park after N µs) | 100 (launcher) | worker park latency vs spin burn | value never swept (20ms→100µs was the B-era move; 25/50/200 untested post-POOL_DIET) |
| `STRATA_ROUTE_TAIL_SKIP` | generate.cpp:8763-8801, expert_source.cpp:3685, :3831 | **7 = ON by default on CUDA serve** (0.1.42, "+11.5% decode (RTX 5090)" era), auto-off >90% residency | skip missed experts every token routes at rank ≥7 (output-changing) | standing-by-default; its lever SHRANK at 97.7% hits — 8/9 never swept (quality-risk sweeps) |
| `STRATA_TAIL_SKIP_MAX_CACHED` | generate.cpp:8781 | 90 (%) | the residency gate above | never swept |
| `STRATA_STAGER_RING` | prefill.cpp:427-453 | 16 (2..256) | prefill pinned staging ring depth | never swept (T13C #6 adjacent, prefill lane) |
| `STRATA_SPEC_RATIO` | generate.cpp:13237 | off (typicality rescue) | exact ratio test (u < min(1,q/p)) instead of typicality | F-27-era arm; not re-swept on the current stack; output-changing |
| `STRATA_TYPICAL_ACCEPT` | generate.cpp:279-283 | 0.2 (launcher) | the typicality rescue threshold | R5-era value; not re-swept post-DT-0.85/post-TYPICAL_DEVICE |
| `STRATA_L2_SETASIDE_MIB` | generate.cpp:8063-8067, device.cu:376-410 | 60 (launcher) | persisting-L2 window over the GDN state span | 72 arm VOIDED by F-205, never re-run; =0 never tested on this stack (§3 #5) |
| `STRATA_ADAPT_LAG/_PINCAP/_MIN_AGE/_DORMANT`, `STRATA_PREDICT_CREDIT/_BUDGET` | generate.cpp:8864-8866, :1470 | 2/48/2/64, 1.0/2 (launcher) | adaptive-tier admission/prediction policy | B6-era values; never swept since; routing-changing family (2× MMLU gate if armed) |
| `STRATA_BATCH_DECODE_SHARE` | generate.cpp:11801 | 0.5 | decode share during multi-chunk prompt reads | T13B C11 known minor; never swept |
| `STRATA_LOOKUP_CHAIN*` | generate.cpp:13256, :13311 | off | n-gram/lookup drafts riding verify rows | never armed in standing; EV-capped by c_row (§3 note) |
| `STRATA_POOL_UMWAIT*` | pool.cpp:40-43 | off | worker wait via WAITPKG | **dead on Linux** (MSVC-only guard, pool.cpp:42) and Zen5 has no WAITPKG — do not queue |

### 2.2 Standing (25, both launchers unless noted)

ATTN_DAILY, GDN_CHUNKED, FUSE_HITS, POOL_DIET, COHORTS=2, TYPICAL_ACCEPT=0.2, POOL_SPIN_US=100,
KV_PREFETCH=1, ADAPT_LAG=2, DMA_BATCH=1, VERIFY_DEVICE_PLAN=1, PREDICT_CREDIT=1.0, PREDICT_BUDGET=2,
ADAPT_PINCAP=48, ADAPT_MIN_AGE=2, ADAPT_DORMANT=64, L2_SETASIDE_MIB=60, BATCH_MTP=1, BATCH_RESCUE=0,
SPEC_COUPLED=1, PLE_BATCH=1, TYPICAL_DEVICE=1, VERIFY_ASYNC_LAUNCH=1, SPEC_DRAFT_TEMP=0.85,
ROUTE_RESIDENT=0.75, ROUTE_RESIDENT_MTP=1 — plus conc-launcher-only ATTN_LANECELL=1.

### 2.3 Verdict history (do not re-queue without a mechanism change)

DRAFT_UNGATED (F-229 −2.73), GDN_SLOTS (F-216 −4.4 M1), GR_WIDE (F-217 −9.2), GR_TILEV (F-217
neutral), PLAN_ONEPASS (F-225 −6.9, default flipped OFF), PLAN_PARTIAL (F-222 neutral), BATCH_G2
(F-223 corrupt+negative), DENSE_MMQ (F-230 decode-negative, prefill-bits tax), BATCH_MTP_MAX_SLOTS
(F-231 below-bar), LFUSE / FUSE_WAIT / DF_PLE (F-206 negatives), HC_SPLIT (F-200 −15.4 as armed),
ATTN_LANECELL (F-200 + F-214 — see §3 #1), spec-min-p 0.35 (F-228 −1.75), mtp-window 4096 (F-214
−4.1), spec 8 (F-203 VOID → closed by T14C arithmetic), DT 0.85 (F-201 swept on BARE base; peak
location never re-checked on the full stack — §3 #7).

---

## 3. (b) Ranked candidate wins (expected %)/(cost), with mechanism reads

Ordering is by (expected %) × (confidence) / (boots+code cost). Every item states its F-232-grade
mechanism (file:line, read this session), gate class, and a one-line cheap falsifier. Expected
percentages are honest ranges, not accounting identities.

### #1. Separate LANECELL from DAILY+COHORTS on the conc lane (stale interaction) — cost XS, expected 0..+3% conc (possibly negative: the boot IS the answer)
- Mechanism: the conc launcher still exports `STRATA_ATTN_LANECELL=1` (launch-custom-iq3s-8081-conc.sh,
  F-215 lock-in). Its +4.6% conc / −3.5% M1 evidence (F-214, 04:11-04:24) predates F-221's DAILY
  adoption, which replaced the same decode-attention kernel class and took heavy-class attention
  4.09 → 1.96 ms/window (F-222 map) — the mechanism LANECELL attacked is mostly gone. F-233's cohort
  3v3 used the LANECELL launcher as its control, so COHORTS' +24.8% includes LANECELL, but LANECELL's
  own post-DAILY, post-COHORTS contribution has NEVER been measured. LANECELL also fired the F-200-era
  caveat "M1-negative" — under COHORTS the solo path is byte-identical, so only conc/pair are exposed.
- Gate: different-schedule (scores layout changes) → M3b canary + acceptance monitor; MMLU not required
  if M3b flat and construction-preserving (same rule as F-221's class), else 2×.
- Cheap falsifier: ONE conc boot with `STRATA_ATTN_LANECELL=0` vs the standing conc launcher; read the
  batch summary rows/s + acceptance. No code, no rebuild.

### #2. Arm the S26 packed/butterfly decode-GEMV package: `STRATA_Q6_PACKED=1` + `STRATA_TSUM=1` (first boot adds `STRATA_Q6P_SELFTEST=1`) — cost XS boot, expected +0.5..2% M1, more if the T=1 head kernel is off-roofline
- Mechanism (code-read, this session): the packed Q6_K kernel is live on CUDA
  (native_mmvq.cu:1855-2100; repack at load via native_head.cpp:75 for the 248,320×2560 output head;
  intercept at native_mmvq.cu:2392) and TSUM is a **bitwise** warp-sum butterfly inside the
  multi-column MMVQ kernels (:1158-1162, applied :1784/:1917). AMD ships both ON by default with
  measured decode wins (arch_defaults.cpp:46-51, S26). On OUR stack the surface is: the head's
  fallback path at T≥5 (verify.cpp:1868 `native_mmvq` — the IL gate at :1863 covers T 1..4), any dense
  shape the IL tables reject, and (via TSUM) the multi-column fallback kernels. The head line is
  0.43-0.48 ms/window at 63-68% roofline (T13A #6, **known**) — the packed kernel + the selftest's
  paired timings finally answer T13A's open question #3 / F-232's "head mmvq_il-at-T1 microbench" for
  free, on the exact shape, in one boot.
- Cost note: the packed head is a SECOND device copy (+~521 MiB); the auto-sized expert cache absorbs
  it (~250 fewer slots ≈ −2.6% residency) — watch the hit-rate line in the same boot.
- Gate: bit-identical (selftest memcmp 1..4 cols + canary). No MMLU.
- Cheap falsifier: the selftest prints paired timings at nc 1/2/4 on the real shapes; if packed ≤ the
  IL/plain rows, retire without a battery.

### #3. `STRATA_HC_Q8=1` — read the hc line as Q8_0 — cost XS boot / S gate, expected −1..+4% M1 (bimodal, see falsifier)
- Mechanism (code-read): hc_q8_env (native_dense.cpp:45-70) replaces the hc projection reads
  (hc_attn/ffn up/down/inject + output_hc_up/down, :71-78) with Q8_0 copies quantized at load
  (:102-139, ggml reference rounding) — half the bytes of bf16 on the CURRENT top kernel line (hc
  reads 2.3-5.2 ms/window, F-218/prof221). Precedent in-file: auto-ON for RDNA4 with "decode +2.5 to
  +4%" measured there (:48, :62-63). Counter-evidence in-file: T12-MEM's read of the line as
  ALU/cvt-bound rather than byte-bound (**known**) — Q8_0 also shrinks the unpack work, so the
  mechanism is not bytes-only, but the honest expectation is bimodal.
- Gate: DIFFERENT BITS (RDNA4 KL 0.0003-0.0010) → M3b + 2× MMLU-Pro. **Check the F-230 tax first**:
  confirm whether the prefill path reads the same replaced tensors (if prefill bits shift, price the
  ~−2.5% acceptance tax in before the battery — one prefill-parity probe answers it).
- Cheap falsifier: one `STRATA_VERIFY_PROFILE` probe with the env on — if the hc0/hc1 down/up stage
  classes don't move, the line is not byte- or unpack-bound and the lane dies for one boot.

### #4. Re-derive the PCIe share at the new hit regime (drop `--pcie-frac` and let PcieAuto pick, or sweep {0.2, 0.5}) — cost XS config, expected 0..+2% (pair/conc favored), bits identical
- Mechanism (code-read): with an explicit `--pcie-frac` the auto-tuner never engages (generate.cpp:674
  "no --pcie-frac: the share is refined"; the tuner PcieAuto at :1821-1866 computes
  f = link/(link + 1.85·r) clamped [0.10, 0.60]). At 97.7%+ hits most miss-layers carry 1-2 misses,
  where `m = (nmiss·pcie_num)>>8` rounds to 0 (expert_source.cpp:3870) — the lane mostly idles and the
  knob's leverage has moved to the tail layers. The 0.35 constant was tuned in the ~30-miss/layer era
  (the stale comment at verify.hpp:561-563 still says "~30 misses/layer"). WHERE a miss computes
  (DMA vs CPU pool) is arithmetic-identical — the frac-0 parity boots (F-222 T1.1) proved bit-parity
  class.
- Cheap falsifier: one boot without `--pcie-frac` — read the PCIe-probe/auto line and the per-request
  pcie_share + hit telemetry already in the logs; if the auto pick lands at 0.35±0.05, the constant is
  fine and the lane closes.

### #5. `STRATA_L2_SETASIDE_MIB` sweep {0, 72} — cost XS, expected −1..+3% M1, bits identical
- Mechanism (code-read): the carve pins a persisting window over the GDN state span
  (generate.cpp:8063-8067 → device.cu:376-410, hitRatio = window/span). Span = 36 layers ×
  (128·48·128 + conv) floats ≈ 110-112 MiB (session.cpp:44-52) vs a 60 MiB window = 54% coverage at
  hitRatio 0.54. F-201's correction says L2 is 96 MB and the docs formula allows ≤72. The F-200 "L2
  72" arm read ≤±2.5% on the BARE base — a verdict F-205 VOIDed, and no re-run ever landed; nor was
  =0 ever tested on the CURRENT stack, where DAILY's KV-reading attention kernel and the expert
  streamers compete for the same L2 the carve takes away (the F-201 µarch brief's "window>carve @1.0
  = 11% SLOWER, measured" warns the wrong policy is real cost, and T13C's brief recorded "L2 carveout
  measured WEAK" externally).
- Gate: bit-identical (cache policy only) → canary only.
- Cheap falsifier: the boot itself — three env values through battery-config.sh, one afternoon.

### #6. `STRATA_PCIE_MIN1=1` — cost XS, expected +0..1% (conc/pair favored), bits identical
- Mechanism (code-read): at ≥2 misses/layer the share rounds to 0 and the misses all queue on the CPU
  pool's serial barrier (expert_source.cpp:3870-3875); MIN1 gives the overlapped DMA lane one blob
  (the fetch checks the pin; unpinned falls back exactly as before). At 97.7%+ hits the typical window
  has a handful of 1-2-miss layers — exactly the shape the knob was written for and never tested on.
- Cheap falsifier: one profile boot; the pool `ms_pool`/`ms_wait` accumulators (verify.cpp:2468-2469)
  and the waitB class read the delta directly.

### #7. DT re-sweep {0.80, 0.90} on the full stack — cost XS (2 config arms), expected 0..+2% M1, proposals-only (no MMLU class)
- Mechanism: F-205's own ledger line: "F-201's DT sweep curve (bare-based) remains valid as a
  dose-response; the peak location may shift on the full base — optional re-sweep." Never done. The
  bare curve was flat-ish near peak (0.80: +11.6 vs 0.85: +13.4), and the full stack added
  TYPICAL_DEVICE rescue + different acceptance state — the optimum plausibly moved.
- Gate: proposals-only (F-201's construction; exact-match verify emits the target's picks) → canary.
- Cheap falsifier: two arm boots vs the standing control; if neither clears +2% on M1, keep 0.85.

### #8. Second sampler sync elision — cost S (code), expected +1..2 t/s, bits identical — **known (T13A #8), still unexecuted, re-confirmed in this read**
- Mechanism (code-read, current line numbers): sampled requests pay the window sync at
  verify.cpp:2486, then launch the sampler and sync AGAIN at :2537 ("m_out_ is the mapped h_out_").
  When `head_sampling_ && (sampled || …)`, the :2486 sync can be skipped — the sampler launch is on
  the same stream and :2537 waits for window + sampler together (the copy_ sync at :2497 and the
  flag protocol are untouched; the next window's stage-in must not race the sampler — one fence
  ordering check).
- Cheap falsifier: STRATA_DECODE_TIMING stage column, greedy vs sampled on the same prompt.

### #9. GDN decode-step ILP rewrite — cost M (kernel), expected +1..2.5% M1 and a c_row cut — designed (F-201 R27), never built
- Mechanism (code-read): the decode recurrence is the per-token walk gdn_step_norm_multi
  (verify.cpp:1246-1257); its kernel shape is documented in its own header as latency-bound — "walks
  i in 0..S-1 TWICE per head with a strided global load each step" (gdn.cu:17-32; the MAX_H=1
  occupancy fix landed, the ILP rewrite did not). F-218's map: rec 0.28 ms light / 0.70 heavy —
  halving is +1.2-2.5% M1 AND shrinks the T-scaled row cost c_row (T14C's flip lever: c_row→2.8
  makes the min-p 0.35 gate-tune positive without any retrain).
- Gate: bit-identical achievable (per-column arithmetic unchanged, same as the F-216 snapshot's
  parity discipline) — canary + the gdn_rec_parity harness. F-217 LAW applies: a fresh nsys on the
  rec line BEFORE kernel surgery.
- Cheap falsifier: microbench gdn_step_norm_multi at our exact (S=128, h_v=48, T=1..8) shapes vs a
  two-pass-interleaved variant, offline — no server boot needed.

### #10. Closed by arithmetic this session (do not spend boots)
- `--spec 4` / `--spec 8` at c_row=3.56: spec-8 was VOID by F-205 but T14C's model + F-228's offer
  distribution (chain trims at ~2.2 by the drafter's own probs) put its EV at ≈ −0.1..−0.3%; spec-4
  trades −2.5% tokens for −2.3% window ≈ neutral. Both stay dead until the FastMTP retrain changes
  the marginals (T14C G4/G5 already schedule the re-tests).
- `--conversation-cache-mib 49152` re-sizing: box has 192 GB RAM (/proc/meminfo this session); pool
  ~40 GB pinned + 48 GB cache + page cache coexist — no pressure story. Not a candidate.
- `--kv-resident 65536`: decode is flat vs depth (kv_stream resolve in-graph; T13C §1.5, **known**);
  doubling only buys VRAM pressure. Dead.
- Lookup-chain drafting (`STRATA_LOOKUP_CHAIN*`, never armed): chain rows pay the same 3.56 ms/row
  c_row (T14C); n-gram marginal acceptance would need ≥0.72 tokens/row — implausible on chat
  traffic. Dead as a speed arm.
- `STRATA_ROUTE_RESIDENT {0.65, 0.85}`, ADAPT_*/PREDICT_* sweeps: **known** (T13C #5) and still
  reasonable one-boots, but routing-changing (2× MMLU each) with sub-±2% expected — inventory, not
  ranking.

---

## 4. (c) Bugs and stale defaults found

1. **Stale constant + comment, verify.hpp:561-563**: `kStagingBlobs = 32` justified by "at
   --pcie-frac 0.35 and ~30 misses/layer the DMA pick wants ~16+ blobs" — the standing boot runs
   97.7%+ hits (~0.2-0.7 misses/layer; the F-233-era logs confirm). Not a speed bug (VRAM only:
   ~75 MB staging that could return ~37 MB to expert slots at 16), but the comment documents a
   vanished regime and the cap has never been revisited downward.
2. **LANECELL staleness** — §3 #1. A committed launcher flag whose supporting evidence describes a
   kernel the stack no longer runs. Not a correctness bug; a measurement-debt bug.
3. **The second sampler sync (verify.cpp:2486 + :2537)** — §3 #8: a real double-wait on sampled
   traffic, sized +1-2 t/s by T13A and never executed.
4. **PcieAuto unengaged by construction** — the engine ships a measured-share auto-tuner
   (generate.cpp:1821-1866) that the standing config permanently disables by passing an explicit
   `--pcie-frac`; note the tuner also self-disables when hits are so high it can't collect 300 jobs —
   so at 98%+ the answer "0.35 is fine" is likely, but it is currently UNMEASURED, not tuned (§3 #4).
5. **arch_defaults.cpp never consulted on CUDA** (comment at :19-20): every S23/S25/S26 decode
   default was validated on gfx1151/gfx12 only. Anything in that table is by-construction untested on
   this rig — the seam §3 #2/#3 mines. Conversely, CUDA has NO equivalent default table; the 5090 runs
   stock kernel dispatch everywhere the launcher doesn't name an env.
6. **F-223's G2×PCIe corruption boundary** was never run to root (banked as characterized). COHORTS
   made the G2 path moot, but the same batch×PCIe interaction surface (device-alias/flagB per-group)
   is still live under COHORTS with pcie 0.35 — F-233's per-verifier buffers removed the known
   trigger, not the diagnosed cause. Watch-class, not action.
7. **F-219's #2758 port** fixed the QFUSE latent self-commit mismatch — verified in-file
   (verify.cpp:884 vs :2705 now both carry `!g_qfuse()`; standing QFUSE-off keeps it latent-safe).
   No open correctness debt found in the sites read this session (draft graphs cached per shape
   mtp.cpp:1311-1313/:1361-1363; min-p trim host-side consistent between the gated and ungated
   readouts :1707-1716 vs :1725; the typicality rescue's emitted-vs-committed overwrite fix at
   generate.cpp:13245-13252 intact).

---

## 5. Contradictions and their resolution

1. **"hc line is bandwidth-bound" (HC_Q8's premise) vs "hc line is ALU/cvt-bound" (T12-MEM, known).**
   Unresolved by literature — resolved empirically by §3 #3's one-probe falsifier (profile the stage
   classes with the env on). Both sources are ours; the boot is cheaper than the argument.
2. **F-200's "L2 72 neutral" vs F-205's void.** The verdict is formally dead (bare base); T13C's
   external "carveout measured WEAK" points the same way but was not our card. Treated as UNTESTED
   (§3 #5), not as neutral.
3. **T13A #6 "IL gated T≥2" vs the current code**: the gate at verify.cpp:1863-1867 now dispatches IL
   at T=1 too (ncols≥1; the `T >= 2` guard only skips the interleave copy). T13A's figure was true at
   its commit; the F-232 queue item "head mmvq_il-at-T1 microbench" is therefore about kernel CHOICE
   at T=1 (Il1Rows vs plain vs packed), not about enabling a path — §3 #2's selftest measures exactly
   that.
4. **"spec-8 rejected" (F-203) vs "F-203 VOID" (F-205)**: never re-run; closed here by T14C
   arithmetic rather than by a boot (§3 #10) — the two independent models (T14C c_row + F-228 offer
   distribution) agree with the voided measurement's sign, so a re-boot buys nothing until the
   drafter changes.

## 6. Open questions

- Does the packed-Q6 repack fit beside the cohort arenas? (boot line prints the MiB; watch free VRAM
  + hit rate in the same boot.)
- Does HC_Q8's load-time replacement touch prefill's read path (the F-230 tax question)? One prefill
  parity probe or a code trace of the WeightRef consumers answers it before any battery.
- What does the conc lane's LANECELL=0 boot read? (§3 #1 — one boot.)
- Whether PcieAuto can even collect at 98% hits (acc_jobs ≥ 300 within its 48-96-window patience) —
  the §3 #4 boot answers it from the engine's own line.
- The GDN rec line's current per-kernel time on sm_120 (no nsys since F-218's phase map) — the §3 #9
  prerequisite.

## 7. Implications

- The cheapest remaining tranche is ENV-ONLY: #1, #2, #4, #5, #6, #7 sum to six boots and no code,
  with two bit-identical-canary gates and one proposals-only sweep. A realistic outcome is +1-4% on
  M1 and +0-3% on conc/pair — not the 250-single goal, but the last cheap percentile band on the
  board before the FastMTP retrain and the kernel tranche.
- The two code items worth their cost are #8 (an afternoon, bit-identical) and #9 (the only item here
  that cuts c_row, which compounds with everything T14C models).
- Nothing found violates a hard law by existing: every candidate is decode-side or policy-side; none
  touches prefill bits (HC_Q8's possible exception is explicitly gated on the §6 check).

## 8. Sources (all read this session unless marked)

- Code: launch-custom-iq3s-8081.sh; launch-custom-iq3s-8081-conc.sh; custom-iq3_s-8081.json;
  src/program/generate.cpp (:276-350 typical/ratio rescue, :674/:1821-1866 PcieAuto, :8763-8801
  tail-skip, :11801 decode share, :12465-13100 pipelined loop, :13170-13400 serial loop);
  src/core/mtp.cpp (:1562-1775 draft + gated/ungated twins, :1311-1387 cached graph capture);
  src/core/verify.cpp (:159 mmvq_il default, :1863-1868 head dispatch, :2440-2543 service tail +
  sampler double-sync, :3089-3220 batch service loop); src/core/native_dense.cpp (:40-160 HC_Q8 +
  Q8_PACKED plumbing); src/core/arch_defaults.cpp (whole file); src/core/session.cpp (:40-52 GDN span
  math); src/core/device.cu (:376-420 setaside arm); src/kernels/cuda/native_mmvq.cu (:1158-1162 TSUM,
  :1729/1855/1985-2100 packed Q8/Q6 + selftest, :2603-2650 IL tables, :2773 supported());
  src/kernels/cuda/gdn.cu (:1-60 kernel shape doc); src/kernels/cuda/sampler.cu (:1262-1266
  ARGMAX_MULTI); src/kernels/cpu/pool.cpp (:40-43 UMWAIT, :478 park); src/core/expert_source.cpp
  (:3678-3688, :3828-3876 tail-skip + pcie share + MIN1); include/strata/core/verify.hpp (:561-563
  staging cap); src/prefill/prefill.cpp (:422-541 stager ring); /proc/meminfo (192 GB).
- Campaign record (known-baseline): SVG-CUSTOM.md F-200..F-233; campaign/research/T13A/B/C-*.md,
  T14A/B/C-*.md, R2-S3-two-cohort-impl-plan.md; the five-boot scoreboard entry.
- Methods note (unproductive searches, not repeated): the pipelined loop at generate.cpp:12465+ is
  multi-GPU only (pipe auto-on requires multi_gpu, :3607) — the standing M1 path is the serial loop at
  :13170-13400; T13A's line numbers have drifted ~+360 (F-233 cohort commit), current anchors are in
  §3. STRATA_GR_FAST/GR_NORM_FAST/PA_FAST had no CUDA decode dispatch found (prefill/HIP-side or
  dead); HC_UPMIX/SELECT_WMMA/WMMA_GEMM are HIP-only; HCD_EXACT needs the gfx1151 hipBLASLt table
  (does nothing on CUDA, arch_defaults comment). The 417-knob grep's long tail is debug/diag (DBG_,
  DUMP_, TEST_, TRACE_ families) — inventoried in the appendix without one-liners.

## Appendix — the full knob inventory (417 names, 549 sites), status-tagged where known

Tags: [STANDING] = in the launcher; other bracketed text = sweep/verdict history from the journal
(this session's annotations); no tag = never set on this rig in any banked boot I could find, function
readable at its site. Debug/diag families (DBG_/DUMP_/TEST_/TRACE_/PARITY) are inventory-only.

- `STRATA_ADAPT_CALLBACK_WARMUP` — src/program/generate.cpp:5132
- `STRATA_ADAPT_DORMANT` — src/program/generate.cpp:8866  [STANDING]
- `STRATA_ADAPT_JOB_CPU` — src/program/generate.cpp:1546
- `STRATA_ADAPT_LAG` — src/program/generate.cpp:1470  [STANDING]
- `STRATA_ADAPT_MIN_AGE` — src/program/generate.cpp:8865  [STANDING]
- `STRATA_ADAPT_NOWAIT` — src/program/generate.cpp:1460
- `STRATA_ADAPT_OVERLAP_RESIDENT` — src/program/generate.cpp:8920
- `STRATA_ADAPT_OVERLAP_RESIDENT` — src/program/generate.cpp:8921
- `STRATA_ADAPT_OVERLAP_SLOW_US` — src/program/generate.cpp:9119
- `STRATA_ADAPT_OVERLAP` — src/program/generate.cpp:1482
- `STRATA_ADAPT_OVERLAP` — src/program/generate.cpp:8964
- `STRATA_ADAPT_PINCAP` — src/program/generate.cpp:8864  [STANDING]
- `STRATA_ADAPT_SIGNAL` — src/program/generate.cpp:8935
- `STRATA_ARENA_LOCK` — src/core/pinned.cu:460
- `STRATA_ARENA_LOCK` — src/core/pinned.cu:488
- `STRATA_ARENA_LOCK` — src/core/pinned.cu:572
- `STRATA_ARENA_LOCK` — src/core/pinned.cu:618
- `STRATA_ARENA_MMAP` — src/core/expert_source.cpp:4617
- `STRATA_ARENA_MMAP` — src/program/generate.cpp:180
- `STRATA_ARENA_MMAP` — src/program/generate.cpp:1911
- `STRATA_ARENA_PIN_GIB` — src/core/pinned.cu:346
- `STRATA_ARENA_RELEASE` — src/program/generate.cpp:6542
- `STRATA_ARGMAX_MULTI` — src/kernels/cuda/sampler.cu:1264  [default ON (sampler.cu:1263)]
- `STRATA_ATTN_DAILY` — src/kernels/cuda/qsa_decode_attn.cu:670  [STANDING]
- `STRATA_ATTN_LANECELL` — src/kernels/cuda/qsa_decode_attn.cu:770  [swept (F-200, F-214) — conc-launcher only; post-DAILY/cohort contribution UNMEASURED (T15C #1)]
- `STRATA_ATTN_MERGE_V2` — src/kernels/cuda/qsa_decode_attn.cu:734  [landed with F-221 (bounded merge)]
- `STRATA_ATTN_PRE75` — src/kernels/cuda/qsa_decode_attn.cu:683  [the T1.2a pre-75 kernel selector (F-221's DAILY selects it)]
- `STRATA_AUX_CPUS` — src/program/generate.cpp:2396
- `STRATA_AUX_STAGER` — src/prefill/prefill.cpp:473
- `STRATA_BATCH_DECODE_SHARE` — src/program/generate.cpp:12108  [default 0.5 (T13B C11 known minor)]
- `STRATA_BATCH_G2` — src/core/verify.cpp:917  [swept NEGATIVE+corrupt (F-223) — characterized boundary]
- `STRATA_BATCH_MTP_MAX_SLOTS` — src/program/generate.cpp:10182  [swept below-bar (F-231, +1.81% conc / pair flag) — instrument]
- `STRATA_BATCH_MTP_SERIAL` — src/program/generate.cpp:4581
- `STRATA_BATCH_MTP` — src/program/generate.cpp:4584  [STANDING]
- `STRATA_BATCH_RESCUE` — src/program/generate.cpp:289  [STANDING]
- `STRATA_BATCH_YIELD_CHUNKS` — src/program/generate.cpp:12124  [default 8]
- `STRATA_BENCH_QMIXED` — src/kernels/fused_gr_bench.cpp:225
- `STRATA_BF16_TC` — src/prefill/gemm.cu:532
- `STRATA_CACHE_MESSAGE_BOUNDARY` — src/program/generate.cpp:12282
- `STRATA_CKPT_REREAD` — src/program/generate.cpp:11714
- `STRATA_CKPT_REREAD` — src/program/generate.cpp:11772
- `STRATA_COHORTS` — src/program/generate.cpp:8110  [STANDING]
- `STRATA_COMMIT_BATCH` — src/core/verify.cpp:2097
- `STRATA_COMMIT_SYNC` — src/core/verify.cpp:2709
- `STRATA_COMMIT_TIMING` — src/core/verify.cpp:2734
- `STRATA_COPY_HIP_PRIO` — src/core/verify.cpp:794
- `STRATA_CPU_YMM` — src/kernels/cpu/expert.cpp:151
- `STRATA_CVEC_FUSE_CHECK` — src/prefill/prefill.cpp:4103
- `STRATA_CVEC_FUSE` — src/prefill/prefill.cpp:362
- `STRATA_DBG_CPU_GATE` — src/prefill/prefill.cpp:3231
- `STRATA_DBG_FEAT_LAYERS` — src/prefill/prefill.cpp:2958
- `STRATA_DBG_FEAT` — src/prefill/prefill.cpp:2952
- `STRATA_DBG_FORCE_SLOW_LAYER` — src/prefill/prefill.cpp:3047
- `STRATA_DBG_GDN` — src/kernels/cuda/verify_kernels.cu:1038
- `STRATA_DBG_GDN` — src/kernels/cuda/verify_kernels.cu:1115
- `STRATA_DBG_NAN` — src/core/verify.cpp:2544
- `STRATA_DBG_NAN` — src/prefill/gemm.cu:905
- `STRATA_DBG_NAN` — src/prefill/prefill.cpp:4058
- `STRATA_DBG_NAN` — src/prefill/prefill.cpp:4232
- `STRATA_DBG_ROUTE_OVERRIDE` — src/prefill/prefill.cpp:2953
- `STRATA_DBG_ROUTE_OVERRIDE` — src/prefill/prefill.cpp:2968
- `STRATA_DEC_BATCH` — src/core/expert_source.cpp:3976
- `STRATA_DEC_BATCH` — src/core/verify.cpp:921
- `STRATA_DEC_BATCH` — src/kernels/cuda/shared_expert.cu:246
- `STRATA_DECODE_TIMING` — src/program/generate.cpp:12433  [diagnostic (never banked on standing boots — T13A)]
- `STRATA_DEFERRED_REGISTER` — src/core/expert_source.cpp:4750
- `STRATA_DENSE_F32` — src/prefill/gemm.cu:580
- `STRATA_DENSE_MMQ_MAXT` — src/prefill/gemm.cu:840  [F-230 T-gate]
- `STRATA_DENSE_MMQ` — src/prefill/gemm.cu:833  [swept NEGATIVE decode (F-230, prefill-bits tax) — instrument kept]
- `STRATA_DEQUANT_WIDE` — src/kernels/cuda/dequant_bf16.cu:269
- `STRATA_DF_BRANCH` — src/core/verify.cpp:820
- `STRATA_DF_PDL` — src/kernels/cuda/verify_kernels.cu:1584
- `STRATA_DF_PDL` — src/kernels/cuda/verify_kernels.cu:1626
- `STRATA_DF_PDL` — src/kernels/pdl_parity.cpp:252
- `STRATA_DF_PLE` — src/core/verify.cpp:1083  [swept NEGATIVE (F-206)]
- `STRATA_DF_QB_Q4` — src/core/verify.cpp:1301
- `STRATA_DISJOINT_ADAPT` — src/program/generate.cpp:5887
- `STRATA_DMA_BATCH` — include/strata/core/dma_batch.hpp:32  [STANDING]
- `STRATA_DRAFT_TIMING` — src/core/mtp.cpp:1539  [diagnostic (g_dt; never banked — T13A open q)]
- `STRATA_DRAFT_TIMING` — src/prefill/prefill.cpp:1463  [diagnostic (g_dt; never banked — T13A open q)]
- `STRATA_DRAFT_UNGATED` — src/core/mtp.cpp:1684  [swept NEGATIVE (F-229, -2.73%) — instrument]
- `STRATA_DUMMY_NODES` — src/core/verify.cpp:1153
- `STRATA_DUMP_FIRST_LOGITS` — src/program/generate.cpp:13214
- `STRATA_DUMP_FIRST_LOGITS` — src/program/generate.cpp:14457
- `STRATA_DUMP_MOE_INPUT` — src/prefill/prefill.cpp:3363
- `STRATA_DUMP_MOE_LAYER` — src/prefill/prefill.cpp:3365
- `STRATA_EARLY_REMOTE_CONTEXTS` — src/program/generate.cpp:2930
- `STRATA_EMB_REUSE_ACCOUNT` — src/prefill/prefill.cpp:279
- `STRATA_EMULATE_CC` — include/strata/core/emulate.hpp:17
- `STRATA_EXCHANGE_ROTATE` — src/core/expert_source.cpp:3299
- `STRATA_EXPERT_V2K` — src/kernels/cuda/iq_kernels.cu:3672
- `STRATA_EXPERT_V2` — src/kernels/cuda/iq_kernels.cu:3654
- `STRATA_EXP_MODE` — src/kernels/cuda/iq_kernels.cu:3593
- `STRATA_FETCH_PVM` — src/core/expert_source.cpp:1569
- `STRATA_FETCH_THREADS` — src/program/generate.cpp:4924
- `STRATA_FILE_RELEASE` — src/core/expert_source.cpp:2599
- `STRATA_FILL_AHEAD` — src/program/generate.cpp:5565
- `STRATA_FORCE_AVX2` — src/kernels/cpu/expert_layout.cpp:43
- `STRATA_FORCE_AVX2` — src/kernels/cpu/expert_layout.cpp:78
- `STRATA_FORCE_ISA` — src/kernels/cpu/expert_layout.cpp:29
- `STRATA_FS_AHEAD` — src/program/generate.cpp:5910
- `STRATA_FS_SLOTS` — src/core/expert_source.cpp:5256
- `STRATA_FS_SLOTS` — src/program/generate.cpp:5909
- `STRATA_FUSE_CONVAB` — src/core/verify.cpp:276
- `STRATA_FUSED_SWIGLU_Q81` — src/kernels/cuda/shared_expert.cu:207
- `STRATA_FUSE_HEAD_GR` — src/core/mtp.cpp:1097
- `STRATA_FUSE_HEAD_GR` — src/core/verify.cpp:1642
- `STRATA_FUSE_HITS` — src/core/verify.cpp:285  [STANDING]
- `STRATA_FUSE_WAIT` — src/core/verify.cpp:252  [swept NEGATIVE (F-206)]
- `STRATA_FUSE_WAIT` — src/core/verify.cpp:268  [swept NEGATIVE (F-206)]
- `STRATA_GDN_CHUNKED` — src/prefill/gdn_rec_parity.cu:567  [STANDING]
- `STRATA_GDN_CHUNKED` — src/prefill/kernels.cu:1038  [STANDING]
- `STRATA_GDN_CHUNKED` — src/prefill/kernels.cu:1993  [STANDING]
- `STRATA_GDN_COMMIT_SPLIT` — src/kernels/cuda/verify_kernels.cu:1037
- `STRATA_GDN_COMMIT_SPLIT` — src/kernels/cuda/verify_kernels.cu:1111
- `STRATA_GDN_CONVL2` — src/prefill/kernels.cu:1923
- `STRATA_GDN_CONV_SERIAL` — src/prefill/kernels.cu:1919
- `STRATA_GDN_HEAD` — src/prefill/kernels.cu:2012
- `STRATA_GDN_KEYHEAD` — src/prefill/kernels.cu:697
- `STRATA_GDN_NOY` — src/prefill/kernels.cu:1971
- `STRATA_GDN_PIPELINE` — src/prefill/kernels.cu:1989
- `STRATA_GDN_PP` — src/prefill/kernels.cu:1964
- `STRATA_GDN_REC_HEADS` — src/prefill/kernels.cu:1985
- `STRATA_GDN_REC_HEADS` — src/prefill/kernels.cu:2013
- `STRATA_GDN_SLOTS` — src/core/verify.cpp:597  [swept NEGATIVE (F-216, -4.4% M1) — parity-proven instrument]
- `STRATA_GDN_SPLIT` — src/kernels/cuda/verify_kernels.cu:1036
- `STRATA_GDN_SPLIT` — src/kernels/cuda/verify_kernels.cu:1088
- `STRATA_GFX1151_DEFAULTS` — src/core/arch_defaults.cpp:33  [n/a CUDA]
- `STRATA_GFX12_CHUNK` — src/program/generate.cpp:2355
- `STRATA_GFX12_CHUNK` — src/program/generate.cpp:6737
- `STRATA_GFX12_DEFAULTS` — src/core/arch_defaults.cpp:21  [n/a CUDA]
- `STRATA_GFX12_DEFAULTS` — src/core/native_dense.cpp:55  [n/a CUDA]
- `STRATA_GFX12_DEFAULTS` — src/program/generate.cpp:2356  [n/a CUDA]
- `STRATA_GFX906_ATTN_QUERY_SWIZZLE` — src/kernels/cuda/qsa_decode_attn.cu:774
- `STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE` — src/kernels/cuda/qsa_decode_attn.cu:778
- `STRATA_GFX906_ATTN_REDUCE12` — src/kernels/cuda/qsa_decode_attn.cu:802
- `STRATA_GFX906_ATTN_REDUCE12_TRACE` — src/kernels/cuda/qsa_decode_attn.cu:809
- `STRATA_GR_DOWN_MAX4` — src/kernels/cuda/fused_gr.cu:1061
- `STRATA_GR_FAST` — src/kernels/cuda/fused_gr.cu:2004
- `STRATA_GR_NORM_FAST` — src/kernels/cuda/fused_gr.cu:1071
- `STRATA_GROUP_COPY` — src/prefill/prefill.cpp:1169
- `STRATA_GROUPED_PAIR_MIN_HITS` — src/kernels/cuda/s2_expert_grouped.cu:542
- `STRATA_GR_SPLIT` — src/kernels/cuda/fused_gr.cu:2123
- `STRATA_GR_TILEV` — src/kernels/cuda/fused_gr.cu:994  [swept NEUTRAL (F-217)]
- `STRATA_GR_UNFUSED` — src/prefill/prefill.cpp:379
- `STRATA_GR_V3` — src/kernels/cuda/fused_gr.cu:2054  [swept neutral-class (F-200 bare-based — VOID verdict, never re-run)]
- `STRATA_GR_V3` — src/kernels/gr_parity.cpp:342  [swept neutral-class (F-200 bare-based — VOID verdict, never re-run)]
- `STRATA_GR_WIDE` — src/kernels/cuda/fused_gr.cu:1133  [swept NEGATIVE (F-217, -9.2%) — L2-traffic instrument]
- `STRATA_HCD_EXACT` — src/prefill/prefill.cpp:357
- `STRATA_HC_Q8_INJECT` — src/core/native_dense.cpp:388
- `STRATA_HC_Q8` — src/core/native_dense.cpp:51  [NEVER ARMED (CUDA default off; RDNA4 default on, +2.5-4% there — T15C #3)]
- `STRATA_HC_SPLIT` — src/kernels/cuda/fused_gr.cu:1188  [swept NEGATIVE (F-200: -15.4%; HC_SPLIT=0 the bad layout)]
- `STRATA_HC_SPLIT` — src/kernels/cuda/fused_gr.cu:2357  [swept NEGATIVE (F-200: -15.4%; HC_SPLIT=0 the bad layout)]
- `STRATA_HC_UPMIX_CHECK` — src/prefill/prefill.cpp:2643
- `STRATA_HC_UPMIX` — src/prefill/prefill.cpp:339
- `STRATA_HEAD_MIX_MULTI` — src/core/mtp.cpp:1102
- `STRATA_HEAD_MIX_MULTI` — src/core/mtp.cpp:1105
- `STRATA_HEAD_MIX_MULTI` — src/core/verify.cpp:232
- `STRATA_HEAD_MIX_MULTI` — src/core/verify.cpp:238
- `STRATA_HIP_ADAPT_KERNEL_COPY` — src/program/generate.cpp:194
- `STRATA_HIPBLASLT_TUNING` — src/prefill/gemm.cu:192
- `STRATA_HIPBLASLT_VERBOSE` — src/prefill/gemm.cu:147
- `STRATA_HIPBLASLT_VERBOSE` — src/prefill/gemm.cu:238
- `STRATA_HIPBLASLT_VERBOSE` — src/prefill/gemm.cu:318
- `STRATA_HIP_PROMPT_F16` — src/prefill/gemm.cu:364
- `STRATA_HIP_ROUTER_FAST` — src/kernels/cuda/router_top10.cu:429
- `STRATA_HIP_ROUTER_OLD` — src/kernels/cuda/router_top10.cu:431
- `STRATA_HIP_WMMA` — src/kernels/cuda/qsa_prompt_attn.cu:1406
- `STRATA_HIP_WMMA` — src/prefill/prefill.cpp:298
- `STRATA_HIP_WMMA` — src/program/generate.cpp:4848
- `STRATA_HOST_CORE` — src/program/generate.cpp:2389
- `STRATA_IDX_FP16_CHECK` — src/prefill/prefill.cpp:2853
- `STRATA_INDEXER_PER_TOKEN` — src/prefill/prefill.cpp:2798
- `STRATA_IO_PF_AHEAD` — src/core/expert_source.cpp:672
- `STRATA_IO_PF_STAGE` — src/core/expert_source.cpp:674
- `STRATA_IO_PF_STAGE` — src/program/generate.cpp:4974
- `STRATA_IO_PF_THREADS` — src/program/generate.cpp:4964
- `STRATA_IO_PREFETCH_DEPTH` — src/program/generate.cpp:5929
- `STRATA_IO_PREFETCH` — src/program/generate.cpp:4963
- `STRATA_IO_STATS` — src/program/generate.cpp:4990
- `STRATA_IO_THREADS` — src/platform/direct_file.cpp:64
- `STRATA_IQ256_GATHER` — src/kernels/cpu/expert_layout.cpp:212
- `STRATA_IQ3S_MT1` — src/kernels/cpu/native_expert.cpp:138
- `STRATA_IQ_GATHER` — src/kernels/cpu/iq_avx512.cpp:33
- `STRATA_IQ_MT_MIN` — src/kernels/cpu/native_expert.cpp:129
- `STRATA_IQ_MT_MIN` — src/kernels/cpu/native_expert.cpp:238
- `STRATA_IQ_MT_MIN` — src/kernels/native_expert_parity.cpp:550
- `STRATA_IQ_PREFETCH` — src/kernels/cpu/iq_avx1.cpp:138
- `STRATA_IQ_PREFETCH` — src/kernels/cpu/iq_avx2.cpp:185
- `STRATA_IQ_PREFETCH` — src/kernels/cpu/iq_avx512.cpp:37
- `STRATA_IQ_STAGE_GRID_MMVQ` — src/kernels/cuda/iq_kernels.cu:2144
- `STRATA_IQ_STAGE_GRID` — src/kernels/cuda/iq_kernels.cu:2137
- `STRATA_KEEP_MAPPING` — src/program/generate.cpp:7291
- `STRATA_KQ256` — src/kernels/cpu/native_expert.cpp:171
- `STRATA_KQ256` — src/kernels/cpu/native_expert.cpp:239
- `STRATA_KV_GROW_FLOOR` — src/program/generate.cpp:6879
- `STRATA_KV_GROW_HOLD` — src/program/generate.cpp:11693
- `STRATA_KV_GROW_INIT` — src/program/generate.cpp:4499
- `STRATA_KV_GROW` — src/program/generate.cpp:4487
- `STRATA_KV_GROW_STEP` — src/program/generate.cpp:6877
- `STRATA_KV_HOST_DMA` — src/prefill/prefill.cpp:2748
- `STRATA_KV_MAIN_OFF` — src/core/layer.cpp:641
- `STRATA_KV_PREFETCH` — src/prefill/prefill.cpp:2101  [STANDING]
- `STRATA_KV_RING_OFF` — src/core/layer.cpp:640
- `STRATA_KV_ROT` — src/program/generate.cpp:2647
- `STRATA_KV_ROT` — src/program/generate.cpp:8336
- `STRATA_KV_STAGE_OWN` — src/prefill/prefill.cpp:894
- `STRATA_L2_SETASIDE_MIB` — src/program/generate.cpp:8065  [STANDING]
- `STRATA_LAYER_HITS` — src/program/generate.cpp:8853
- `STRATA_LAYER_SPLIT_GATE` — src/program/generate.cpp:4138
- `STRATA_LAYER_SPLIT_PROBE` — src/program/generate.cpp:5721
- `STRATA_LFUSE_GATE` — src/core/verify.cpp:153
- `STRATA_LFUSE_PAIR` — src/core/verify.cpp:154
- `STRATA_LFUSE` — src/core/verify.cpp:90  [swept NEGATIVE (F-206) on CUDA (de05c182 un-gate)]
- `STRATA_LOGPOS_EXTRA` — src/program/generate.cpp:11938
- `STRATA_LOGPOS` — src/program/generate.cpp:11900
- `STRATA_LOGPOS` — src/program/generate.cpp:11934
- `STRATA_LOGPOS_TOPK` — src/core/verify.cpp:2691
- `STRATA_LOOKAHEAD_K` — src/program/generate.cpp:5925
- `STRATA_LOOKAHEAD` — src/program/generate.cpp:5914
- `STRATA_LOOKAHEAD_STATS` — src/program/generate.cpp:5912
- `STRATA_LOOKUP_CHAIN_FIXED` — src/program/generate.cpp:12423  [lookup-chain drafting — never armed in standing; EV-capped by c_row (T15C note)]
- `STRATA_LOOKUP_CHAIN_FIXED` — src/program/generate.cpp:14389  [lookup-chain drafting — never armed in standing; EV-capped by c_row (T15C note)]
- `STRATA_MEMORY_LIMIT_MIB` — src/program/generate.cpp:2407
- `STRATA_MMQ_KQ` — src/prefill/mmq_kq_loader.cpp:110
- `STRATA_MMQ_KQ` — src/prefill/moe_mmq.cu:31
- `STRATA_MMQ_KQ` — src/qwen35/q36_engine.cu:2409
- `STRATA_MMQ_RESIDENT_SORT_NE` — src/prefill/mmq_resident_sort.hpp:17
- `STRATA_MMVF_ROWS` — src/kernels/cuda/native_bf16.cu:272
- `STRATA_MMVF_ROWS` — src/kernels/cuda/native_bf16.cu:302
- `STRATA_MMVQ_IL_ROWS` — src/kernels/cuda/native_mmvq.cu:2678  [runtime table override — never swept here]
- `STRATA_MMVQ_IL` — src/core/verify.cpp:159  [default ON (verify.cpp:159); IL-vs-plain-vs-packed at T=1..4 unmeasured on sm_120 (T15C #2 falsifier)]
- `STRATA_MMVQ_WAVE` — src/kernels/cuda/native_mmvq.cu:1623
- `STRATA_MMVQ_WAVE` — src/kernels/mmvq_multi_parity.cpp:245
- `STRATA_MTP_BATCH_F16` — src/prefill/prefill.cpp:1432
- `STRATA_MTP_BATCH_RING` — src/prefill/prefill.cpp:1396
- `STRATA_MTP_BATCH` — src/prefill/prefill.cpp:1391
- `STRATA_MTP_CATCHUP_ALL` — src/core/mtp.cpp:96
- `STRATA_MTP_FULL_HEAD` — src/core/mtp.cpp:540
- `STRATA_MTP_FULL_HEAD` — src/program/generate.cpp:3595
- `STRATA_MTP_HIP_PRIO` — src/core/mtp.cpp:483
- `STRATA_MTP_PREFILL_SYNC` — src/core/mtp.cpp:1439
- `STRATA_MTP_PRIORITY` — src/core/mtp.cpp:469
- `STRATA_MTP_PROFILE` — src/core/mtp.cpp:801
- `STRATA_MTP_SHARED_BRANCH` — src/core/mtp.cpp:1115
- `STRATA_MTP_SIDE_STREAM` — src/core/mtp.cpp:515
- `STRATA_MTP_STAGE0` — src/program/generate.cpp:3744
- `STRATA_MTP_TAP` — src/core/mtp.cpp:1232
- `STRATA_MTP_TAP` — src/core/mtp.cpp:288
- `STRATA_MTP_TAP` — src/core/mtp.cpp:77
- `STRATA_MTP_TOP2` — src/core/mtp.cpp:546
- `STRATA_MULTI_BLOCK_ARGMAX` — src/kernels/cuda/verify_kernels.cu:942
- `STRATA_NAN_GUARD` — src/core/verify.cpp:3430
- `STRATA_NAN_INJECT` — src/core/verify.cpp:3441
- `STRATA_NO_ARENA_THP` — src/core/pinned.cu:289
- `STRATA_NO_ARENA_THP` — src/core/pinned.cu:294
- `STRATA_NO_AVXVNNI` — src/kernels/cpu/expert_layout.cpp:262
- `STRATA_NO_BATCH_KV_STEP` — src/core/mtp.cpp:974
- `STRATA_NO_IQ128` — src/kernels/cpu/native_expert.cpp:166
- `STRATA_NO_IQ128` — src/kernels/cpu/native_expert.cpp:252
- `STRATA_NO_IQ128` — src/program/generate.cpp:2997
- `STRATA_NO_IQ256` — src/kernels/cpu/native_expert.cpp:158
- `STRATA_NO_IQ256` — src/program/generate.cpp:3000
- `STRATA_NO_IQ4NL` — src/kernels/cpu/native_expert.cpp:237
- `STRATA_NO_IQ512` — src/kernels/cpu/native_expert.cpp:150
- `STRATA_NO_LARGEPAGES` — src/core/pinned.cu:122
- `STRATA_NO_LARGEPAGES` — src/core/pinned.cu:139
- `STRATA_NO_LARGEPAGES` — src/core/pinned.cu:238
- `STRATA_NO_LARGEPAGES` — src/core/pinned.cu:290
- `STRATA_NO_LARGEPAGES` — src/core/pinned.cu:297
- `STRATA_NO_LARGEPAGES` — src/core/pinned.cu:310
- `STRATA_NO_MMVQ_ROWS2` — src/kernels/cuda/native_mmvq.cu:1297
- `STRATA_NO_MMVQ_ROWS2` — src/kernels/cuda/native_mmvq.cu:1324
- `STRATA_NO_MULTI_GR` — src/core/mtp.cpp:1170
- `STRATA_NO_MULTI_GR` — src/kernels/cuda/fused_gr.cu:1114
- `STRATA_NO_MULTI_GR` — src/kernels/cuda/gr.cu:416
- `STRATA_NO_NORM_ROPE` — src/kernels/cuda/native_rope.cu:177
- `STRATA_NO_NVFP4_256` — src/kernels/cpu/native_expert.cpp:213
- `STRATA_NO_NVFP4_256` — src/kernels/cpu/native_expert.cpp:265
- `STRATA_NO_NVFP4_512` — src/kernels/cpu/native_expert.cpp:206
- `STRATA_NO_NVFP4_512` — src/kernels/cpu/native_expert.cpp:260
- `STRATA_NO_Q8K_AVX1` — src/kernels/cpu/native_expert.cpp:107
- `STRATA_NO_Q8K_AVX2` — src/kernels/cpu/native_expert.cpp:101
- `STRATA_NORM_ROPE` — src/kernels/cuda/native_rope.cu:180
- `STRATA_NVFP4_PREFETCH` — src/kernels/cpu/nvfp4_avx2.cpp:51
- `STRATA_NVFP4_PREFETCH` — src/kernels/cpu/nvfp4_avx512.cpp:52
- `STRATA_NVFP4_PREFETCH` — src/kernels/cpu/nvfp4_avx512_parity.cpp:244
- `STRATA_OLD_GROUPED` — src/kernels/cuda/s2_expert_grouped.cu:529
- `STRATA_ONE_TOKEN_COMMIT` — src/core/verify.cpp:305
- `STRATA_ONE_TOKEN_COMMIT` — src/core/verify.cpp:311
- `STRATA_OWNED_PRICE` — src/program/generate.cpp:5193
- `STRATA_PA_FAST` — src/kernels/cuda/qsa_prompt_attn.cu:1433
- `STRATA_PARITY_DUMP` — src/kernels/route_resident_parity.cu:218
- `STRATA_PARTIAL_PIN_GIB` — src/core/expert_source.cpp:2974
- `STRATA_PARTIAL_PIN` — src/core/expert_source.cpp:2968
- `STRATA_PCIE_AUTO_CANON` — src/program/generate.cpp:2960
- `STRATA_PCIE_AUTO_SPLIT` — src/program/generate.cpp:1709
- `STRATA_PCIE_FRAC_DEFAULT` — src/program/generate.cpp:2956
- `STRATA_PCIE_MIN1` — src/core/expert_source.cpp:3874  [NEVER SWEPT (opt-in; T15C #8)]
- `STRATA_PEER_DIRECT` — src/core/peer_experts.cpp:190
- `STRATA_PEER_HOT_AT` — src/program/generate.cpp:3768
- `STRATA_PEER_HOT` — src/program/generate.cpp:3766
- `STRATA_PEER_NO_P2P` — src/core/peer_experts.cpp:101
- `STRATA_PF_BK64` — src/prefill/wmma_gemm.cu:693
- `STRATA_PF_FUSED_KQ` — src/prefill/moe_fused_iq.cu:1071
- `STRATA_PF_FUSED_NATIVE` — src/prefill/moe_fused_iq.cu:1120
- `STRATA_PF_FUSED` — src/prefill/moe_fused.cu:658
- `STRATA_PF_FUSED` — src/prefill/moe_fused.cu:669
- `STRATA_PF_FUSED` — src/prefill/moe_fused_iq.cu:1128
- `STRATA_PF_FUSED_TILE` — src/prefill/moe_fused_iq.cu:1107
- `STRATA_PF_GEMM` — src/prefill/gemm.cu:745
- `STRATA_PF_GEMM` — src/prefill/gemm.cu:928
- `STRATA_PF_HCDOWN` — src/prefill/prefill.cpp:349
- `STRATA_PF_OCC` — src/prefill/moe_fused.cu:589
- `STRATA_PF_OCC` — src/prefill/moe_fused_iq.cu:996
- `STRATA_PF_OCC_W` — src/prefill/moe_fused_iq.cu:1009
- `STRATA_PF_PAD` — src/prefill/gemm.cu:927
- `STRATA_PF_PAD` — src/prefill/prefill.cpp:370
- `STRATA_PF_PEER_COMPACT` — src/prefill/prefill.cpp:1740
- `STRATA_PF_PEER_F16` — src/prefill/prefill.cpp:1785
- `STRATA_PF_PEER_GATHER` — src/prefill/prefill.cpp:1788
- `STRATA_PF_PEER_HOST` — src/prefill/prefill.cpp:1692
- `STRATA_PF_PEER_OUT_PIPE` — src/prefill/prefill.cpp:1724
- `STRATA_PF_PEER_RING` — src/prefill/prefill.cpp:1758
- `STRATA_PF_PEER_STREAM` — src/prefill/prefill.cpp:1754
- `STRATA_PF_PEER_SUMS` — src/prefill/prefill.cpp:1781
- `STRATA_PF_PEER_XSTREAM` — src/prefill/prefill.cpp:1709
- `STRATA_PF_STEP_SYNC` — src/prefill/prefill.cpp:2527
- `STRATA_PF_SWITCH_MIN_T` — src/kernels/cuda/qsa_prompt_attn.cu:1434
- `STRATA_PF_SWITCH_MIN_T` — src/prefill/gemm.cu:746
- `STRATA_PF_SWITCH_MIN_T` — src/prefill/gemm.cu:929
- `STRATA_PF_SWITCH_MIN_T` — src/prefill/prefill.cpp:335
- `STRATA_PIN_BY_NEED_SKIP` — src/program/generate.cpp:5033
- `STRATA_PIN_BY_NEED` — src/program/generate.cpp:5022
- `STRATA_PIN_GUARD` — src/core/expert_source.cpp:2887
- `STRATA_PIN_RESERVE_GIB` — src/core/expert_source.cpp:2880
- `STRATA_PIPELINE_ADAPT_ASYNC` — src/program/generate.cpp:7081
- `STRATA_PIPELINE_AUTO_MIN_STAGES` — src/program/generate.cpp:3603
- `STRATA_PIPELINE_AUTO` — src/program/generate.cpp:3602  [multi-GPU only (generate.cpp:3607) — inert single-GPU]
- `STRATA_PIPELINE_DEBUG` — src/program/generate.cpp:268
- `STRATA_PLAN_DIFF` — src/core/expert_source.cpp:3756  [audit harness (F-224)]
- `STRATA_PLAN_ONEPASS` — src/core/expert_source.cpp:3755  [swept NEGATIVE (F-225, -6.9%) — default flipped OFF]
- `STRATA_PLAN_PARTIAL` — src/core/expert_source.cpp:3868  [swept NEUTRAL (F-222 T1.1)]
- `STRATA_PLAN_PARTIAL` — src/core/verify.cpp:1567  [swept NEUTRAL (F-222 T1.1)]
- `STRATA_PLE_AUTO_MMAP` — src/program/generate.cpp:3417
- `STRATA_PLE_BATCH` — src/core/verify.cpp:1039  [STANDING]
- `STRATA_PLE_BATCH` — src/core/verify.cpp:1048  [STANDING]
- `STRATA_PLE_BATCH` — src/prefill/prefill.cpp:2154  [STANDING]
- `STRATA_PLE_GGUF` — src/kernels/ple_parity.cpp:257
- `STRATA_PLE_PROBE` — src/program/generate.cpp:3406
- `STRATA_POOL_DIET` — src/kernels/cpu/pool.cpp:480  [STANDING]
- `STRATA_POOL_SPIN_US` — src/kernels/cpu/pool.cpp:478  [STANDING]
- `STRATA_POOL_TRACE` — src/core/expert_source.cpp:3723  [diagnostic]
- `STRATA_POOL_UMWAIT_C02` — src/kernels/cpu/pool.cpp:58  [dead on Linux]
- `STRATA_POOL_UMWAIT_ESC` — src/kernels/cpu/pool.cpp:64  [dead on Linux]
- `STRATA_POOL_UMWAIT` — src/kernels/cpu/pool.cpp:40  [dead on Linux (MSVC-only, pool.cpp:42) and no WAITPKG on Zen5]
- `STRATA_PREDICT_BUDGET` — src/program/generate.cpp:9013  [STANDING]
- `STRATA_PREDICT_CREDIT` — src/program/generate.cpp:9010  [STANDING]
- `STRATA_PREFILL_AUTO_MAX` — src/program/generate.cpp:2130
- `STRATA_PREFILL_BF16X2` — src/prefill/prefill.cpp:325
- `STRATA_PREFILL_CPU_SHARE_MAX` — src/prefill/prefill.cpp:203
- `STRATA_PREFILL_CPU_SHARE_MAX` — src/prefill/prefill.cpp:206
- `STRATA_PREFILL_CPU_SHARE` — src/prefill/prefill.cpp:186
- `STRATA_PREFILL_CPU_SHARE` — src/program/generate.cpp:7322
- `STRATA_PREFILL_DUMP_R_ALL` — src/prefill/prefill.cpp:4193
- `STRATA_PREFILL_DUMP_R` — src/prefill/prefill.cpp:4180
- `STRATA_PREFILL_EQUAL` — src/program/generate.cpp:6688
- `STRATA_PREFILL_GROUP_GATHER` — src/prefill/prefill.cpp:3831
- `STRATA_PREFILL_HELPER_SOURCE` — src/program/generate.cpp:7355
- `STRATA_PREFILL_HELP_FRAC` — src/prefill/prefill.cpp:1573
- `STRATA_PREFILL_HELP` — src/prefill/prefill.cpp:1561
- `STRATA_PREFILL_ISSUER` — src/prefill/prefill.cpp:2460
- `STRATA_PREFILL_LEND_PCT` — src/program/generate.cpp:3997
- `STRATA_PREFILL_LEND_PCT` — src/program/generate.cpp:5727
- `STRATA_PREFILL_LEND_PCT` — src/program/generate.cpp:6709
- `STRATA_PREFILL_MMQ` — src/prefill/prefill.cpp:1016
- `STRATA_PREFILL_NVFP4` — src/prefill/mmq_expert_bench.cpp:112
- `STRATA_PREFILL_NVFP4` — src/prefill/mmq_nvfp4_parity.cpp:337
- `STRATA_PREFILL_NVFP4` — src/prefill/moe_mmq.cu:189
- `STRATA_PREFILL_RING` — src/prefill/prefill.cpp:293
- `STRATA_PREFILL_STREAM_AHEAD` — src/prefill/prefill.cpp:127
- `STRATA_PREFILL_STREAM_MIN` — src/prefill/prefill.cpp:165
- `STRATA_PREFILL_TIMING` — src/prefill/prefill.cpp:2012
- `STRATA_PREFILL_TIMING` — src/prefill/prefill.cpp:2048
- `STRATA_PROMPT_ATTN_OLD` — src/prefill/prefill.cpp:2921
- `STRATA_PROMPT_ATTN_Q4` — src/kernels/cuda/qsa_prompt_attn.cu:1500
- `STRATA_PROMPT_ATTN_V1` — src/kernels/cuda/qsa_prompt_attn.cu:1517
- `STRATA_PROMPT_DIRECT` — src/program/generate.cpp:7278
- `STRATA_Q2_BITPLANE` — src/kernels/cpu/q2_avx1.cpp:179
- `STRATA_Q2_BITPLANE` — src/kernels/cpu/q2_avx2.cpp:147
- `STRATA_Q36_BLAS_F32` — src/qwen35/q36_engine.cu:1288
- `STRATA_Q36_FIXED_SLOTS` — src/qwen35/q36_engine.cu:1703
- `STRATA_Q36_KEEP_PACKED` — src/qwen35/q36_engine.cu:1964
- `STRATA_Q36_NO_MMQ` — src/qwen35/q36_engine.cu:1168
- `STRATA_Q36_NO_MTP_CPU` — src/qwen35/q36_engine.cu:1961
- `STRATA_Q36_NO_MTP_CPU` — src/qwen35/q36_engine.cu:1986
- `STRATA_Q36_PLACE_DEBUG` — src/qwen35/q36_engine.cu:2439
- `STRATA_Q36_STREAM_ALL` — src/qwen35/q36_engine.cu:1583
- `STRATA_Q6_PACKED` — src/kernels/cuda/native_mmvq.cu:2027  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_Q6P_GRID` — src/kernels/cuda/native_mmvq.cu:1867
- `STRATA_Q6P_SELFTEST` — src/kernels/cuda/native_mmvq.cu:2065  [built-in bitwise check + paired timings (T15C #2 falsifier)]
- `STRATA_Q8_PACKED` — src/kernels/cuda/native_mmvq.cu:1992  [NEVER ARMED (no eligible Q8_0 shapes in this pack — dense are Q6_K)]
- `STRATA_QFUSE_HC_LOCAL` — src/kernels/cuda/fused_gr.cu:1088
- `STRATA_QFUSE_HC` — src/core/verify.cpp:164
- `STRATA_QFUSE` — src/core/arch_defaults_test.cpp:67  [OFF standing; latent #2758 mismatch FIXED (F-219)]
- `STRATA_QFUSE` — src/core/verify.cpp:176  [OFF standing; latent #2758 mismatch FIXED (F-219)]
- `STRATA_QSA_CLUSTER` — src/kernels/cuda/qsa_select.cu:1308
- `STRATA_QSA_DUMP_LAST` — src/prefill/prefill.cpp:2900
- `STRATA_QSA_DUMP_LAST` — src/prefill/prefill.cpp:2901
- `STRATA_QSA_DUMP` — src/prefill/prefill.cpp:2899
- `STRATA_QSA_EARLY_EXIT` — src/kernels/cuda/qsa_select.cu:1106
- `STRATA_QSA_WARP` — src/kernels/cuda/qsa_prompt_attn.cu:1477
- `STRATA_QSA_WARP` — src/kernels/cuda/qsa_select.cu:1155
- `STRATA_READ_AHEAD` — src/platform/memory.cpp:207
- `STRATA_REFILL_BLOCKING` — src/program/generate.cpp:258
- `STRATA_REFILL_SERIAL` — src/program/generate.cpp:11997
- `STRATA_REMOTE_SPIN` — src/core/remote_experts.cpp:79
- `STRATA_REMOTE_ZEROCOPY` — src/core/remote_experts.cpp:217
- `STRATA_RESIDENT_HEADROOM_GIB` — src/program/generate.cpp:2309
- `STRATA_RESIDENT_HEADROOM_GIB` — src/program/generate.cpp:2320
- `STRATA_RESIDENT_PIN` — src/program/generate.cpp:2307
- `STRATA_RESIDENT_PIN` — src/program/generate.cpp:2318
- `STRATA_RING_BYTES` — src/prefill/prefill.cpp:283
- `STRATA_ROPE_TABLE` — src/kernels/cuda/native_rope.cu:154
- `STRATA_ROUTE_PER_TOKEN` — src/core/layer.cpp:400
- `STRATA_ROUTE_PRIOR` — src/core/verify.cpp:1547
- `STRATA_ROUTE_PROJ_PER_TOKEN` — src/core/layer.cpp:412
- `STRATA_ROUTER_BLOCK` — src/kernels/cuda/router_top10.cu:462
- `STRATA_ROUTE_RESIDENT_MTP` — src/program/generate.cpp:7872  [STANDING]
- `STRATA_ROUTE_RESIDENT_RANKS` — src/core/verify.cpp:147
- `STRATA_ROUTE_RESIDENT_RANKS` — src/program/generate.cpp:7875
- `STRATA_ROUTE_RESIDENT` — src/core/verify.cpp:146  [STANDING]
- `STRATA_ROUTE_RESIDENT` — src/program/generate.cpp:7871  [STANDING]
- `STRATA_ROUTE_RESIDENT` — src/program/generate.cpp:7873  [STANDING]
- `STRATA_ROUTE_TAIL_SKIP` — src/core/expert_source.cpp:3685  [ON by default on CUDA serve since 0.1.42 (generate.cpp:8763; output-changing; gated off >90% residency — our 38% keeps it armed)]
- `STRATA_ROUTE_TAIL_SKIP` — src/program/generate.cpp:8769  [ON by default on CUDA serve since 0.1.42 (generate.cpp:8763; output-changing; gated off >90% residency — our 38% keeps it armed)]
- `STRATA_RSS_TRACE` — src/program/generate.cpp:244
- `STRATA_S2_SWIGLU_Q8` — src/kernels/cuda/s2_expert_grouped.cu:1157
- `STRATA_SCORES_MULTI` — src/kernels/cuda/qsa_select.cu:1103
- `STRATA_SELECT_OLD` — src/prefill/prefill.cpp:2820
- `STRATA_SELECT_SIMT` — src/kernels/cuda/qsa_select.cu:1162
- `STRATA_SELECT_WMMA` — src/kernels/cuda/qsa_select.cu:1129
- `STRATA_SEL_OVERLAP` — src/prefill/prefill.cpp:2831
- `STRATA_SESSION_BUFFERED` — src/core/conversation_file.cpp:340
- `STRATA_SESSION_BUFFERED` — src/core/conversation_file_test.cpp:209
- `STRATA_SESSION_GOLDEN_PRINT` — src/core/conversation_file_test.cpp:180
- `STRATA_SH_FORK_LATE` — src/core/verify.cpp:1502
- `STRATA_SH_STREAM` — src/core/verify.cpp:214
- `STRATA_SM60_TABLE` — src/kernels/cuda/iq_kernels.cu:3611
- `STRATA_SM60_TABLE` — src/kernels/cuda/native_mmvq.cu:2753
- `STRATA_SM61_TABLE` — src/kernels/cuda/iq_kernels.cu:3616
- `STRATA_SM61_TABLE` — src/kernels/cuda/native_mmvq.cu:2756
- `STRATA_SM70_TABLE` — src/kernels/cuda/fused_gr.cu:2010
- `STRATA_SM70_TABLE` — src/kernels/cuda/iq_kernels.cu:3608
- `STRATA_SM70_TABLE` — src/kernels/cuda/native_mmvq.cu:2750
- `STRATA_SNAPSHOT_FULL_CAPTURE` — src/program/generate.cpp:11659
- `STRATA_SNAPSHOT_VERIFY` — src/program/generate.cpp:11632
- `STRATA_SNAPSHOT_VERIFY` — src/program/generate.cpp:8678
- `STRATA_SPEC_COUPLED` — include/strata/core/coupled_draft.hpp:59  [STANDING]
- `STRATA_SPEC_DEPTH` — src/program/generate.cpp:13716  [diagnostic (F-228 banked the stair)]
- `STRATA_SPEC_DRAFT_TEMP` — src/kernels/cuda/sampler.cu:1329  [STANDING]
- `STRATA_SPEC_MIN_TEMP` — include/strata/core/spec_prob.hpp:69  [not swept]
- `STRATA_SPEC_PROB_DT` — include/strata/core/spec_prob.hpp:57
- `STRATA_SPEC_PROB_GATE` — include/strata/core/spec_prob.hpp:47
- `STRATA_SPEC_PROB` — include/strata/core/coupled_draft.hpp:69
- `STRATA_SPEC_RATIO` — src/program/generate.cpp:10325  [F-27-era arm (ratio vs typicality rescue); not re-swept on current stack]
- `STRATA_SPEC_RATIO` — src/program/generate.cpp:10538  [F-27-era arm (ratio vs typicality rescue); not re-swept on current stack]
- `STRATA_SPEC_RATIO` — src/program/generate.cpp:13237  [F-27-era arm (ratio vs typicality rescue); not re-swept on current stack]
- `STRATA_SPLIT_CHUNK16K` — src/program/generate.cpp:6722
- `STRATA_SPLIT_COVER_B` — src/program/generate.cpp:3975
- `STRATA_SPLIT_COVER_B` — src/program/generate.cpp:3981
- `STRATA_SPLIT_LINK_ABS` — src/program/generate.cpp:3941
- `STRATA_SPLIT_MISS_MS` — src/program/generate.cpp:3949
- `STRATA_SPLIT_MTP_BATCH` — src/program/generate.cpp:8652
- `STRATA_SPLIT_NEAR_TIE_PCT` — src/program/generate.cpp:4395
- `STRATA_SPLIT_OWN_BUFFERS` — src/program/generate.cpp:7149
- `STRATA_SPLIT_OWN_BUFFERS` — src/program/generate.cpp:7591
- `STRATA_SPLIT_OWN_PLACE` — src/program/generate.cpp:3891
- `STRATA_SPLIT_OWN` — src/program/generate.cpp:2547
- `STRATA_SPLIT_RING` — src/program/generate.cpp:7107
- `STRATA_SPLIT_SMALL_MAX` — src/program/generate.cpp:7825
- `STRATA_SPLIT_SMALL_OWN` — src/program/generate.cpp:7823
- `STRATA_SPLIT_TIMING` — src/program/generate.cpp:13687
- `STRATA_SSD_KEEPALIVE` — src/program/generate.cpp:3394
- `STRATA_SSD_KEEPALIVE_WINDOW` — src/program/generate.cpp:3395
- `STRATA_STAGE_PIN` — src/core/expert_source.cpp:1293
- `STRATA_STAGER_BATCH` — src/prefill/prefill.cpp:1157
- `STRATA_STAGER_RING` — src/prefill/prefill.cpp:1153  [prefill stager ring, default 16 — never swept (T13C #6 adj)]
- `STRATA_STAGER_RING` — src/prefill/prefill.cpp:453  [prefill stager ring, default 16 — never swept (T13C #6 adj)]
- `STRATA_STAGER_SLEEP` — src/prefill/prefill.cpp:147
- `STRATA_STAGER_SSD` — src/prefill/prefill.cpp:1145
- `STRATA_STAGER_THREADS` — src/prefill/prefill.cpp:1133
- `STRATA_STAGE_TRIM` — src/program/generate.cpp:3167
- `STRATA_STARTUP_UNBUFFERED` — src/core/expert_source.cpp:1780
- `STRATA_STATE_HASH_GDN` — src/prefill/prefill.cpp:4300
- `STRATA_STATE_HASH_GDN` — src/program/generate.cpp:13441
- `STRATA_STATE_HASH` — src/program/generate.cpp:13403
- `STRATA_TAIL_SKIP_MAX_CACHED` — src/program/generate.cpp:8781  [default 90 (generate.cpp:8781)]
- `STRATA_TEST_CACHE_FAIL` — src/program/generate.cpp:5460
- `STRATA_TEST_PAGEABLE` — include/strata/core/pinned_ring.hpp:138
- `STRATA_TEST_PAGEABLE` — src/prefill/prefill.cpp:138
- `STRATA_TEST_PINNED_SHARE` — src/program/generate.cpp:6645
- `STRATA_TEST_SLOW_COPY_US` — src/core/conversation_copy.hpp:49
- `STRATA_TEST_UNMAP_GGUF` — src/core/expert_source.cpp:1761
- `STRATA_TEST_VERIFY_STALL` — src/core/verify.cpp:393
- `STRATA_TG_FLUSH_US` — src/core/session.cpp:941
- `STRATA_TOPK_ACTIVE_ANY` — src/kernels/cuda/qsa_select.cu:1326
- `STRATA_TOPK_CAPACITY_GUARD` — src/kernels/cuda/qsa_select.cu:1323
- `STRATA_TOPK_OLD` — src/kernels/cuda/qsa_select.cu:1302
- `STRATA_TOPK_STREAM` — src/kernels/cuda/qsa_select.cu:1353
- `STRATA_TRACE_ADAPT` — src/program/generate.cpp:14197
- `STRATA_TRACE_ADAPT` — src/program/generate.cpp:14227
- `STRATA_TRACE` — src/prefill/prefill.cpp:2180
- `STRATA_TRACE` — src/program/generate.cpp:1441
- `STRATA_TRACE` — src/program/generate.cpp:9712
- `STRATA_TSUM` — src/core/arch_defaults_test.cpp:66  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_TSUM` — src/kernels/cuda/fused_gr.cu:1970  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_TSUM` — src/kernels/cuda/iq_kernels.cu:3657  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_TSUM` — src/kernels/cuda/iq_kernels.cu:3687  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_TSUM` — src/kernels/cuda/native_bf16.cu:236  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_TSUM` — src/kernels/cuda/native_mmvq.cu:1160  [NEVER ARMED on this rig (T15C #2)]
- `STRATA_TYPICAL_ACCEPT` — src/program/generate.cpp:280  [STANDING]
- `STRATA_TYPICAL_DEVICE` — src/program/generate.cpp:300  [STANDING]
- `STRATA_UMA_HEADROOM_GIB` — src/core/expert_cache.cpp:55
- `STRATA_UNBUFFERED_LOAD` — src/core/expert_source.cpp:1693
- `STRATA_UNBUFFERED_LOAD` — src/core/expert_source.cpp:519
- `STRATA_UNBUFFERED_LOAD` — src/core/pinned.cu:739
- `STRATA_UNBUFFERED_LOAD` — src/program/generate.cpp:4943
- `STRATA_VERIFY_ALL_RESIDENT` — src/core/verify.cpp:838
- `STRATA_VERIFY_ARENA` — src/core/expert_source.cpp:4816
- `STRATA_VERIFY_ASYNC_LAUNCH` — src/core/verify.cpp:2345  [STANDING]
- `STRATA_VERIFY_COMPLEMENT` — src/core/expert_source.cpp:3196
- `STRATA_VERIFY_COMPLEMENT` — src/program/generate.cpp:5610
- `STRATA_VERIFY_DEBUG` — src/core/verify.cpp:89
- `STRATA_VERIFY_DEVICE_PLAN` — src/core/expert_source.cpp:3979  [STANDING]
- `STRATA_VERIFY_DEVICE_PLAN` — src/core/expert_source.cpp:5259  [STANDING]
- `STRATA_VERIFY_DEVICE_PLAN` — src/core/verify.cpp:855  [STANDING]
- `STRATA_VERIFY_NODES` — src/core/mtp.cpp:852
- `STRATA_VERIFY_NODES` — src/core/verify.cpp:1965
- `STRATA_VERIFY_NODES` — src/core/verify.cpp:1993
- `STRATA_VERIFY_NODES` — src/core/verify.cpp:2004
- `STRATA_VERIFY_PROFILE` — src/core/verify.cpp:769  [diagnostic (F-218/F-222 maps)]
- `STRATA_VERIFY_QDEDUP` — src/core/verify.cpp:155
- `STRATA_WATCHDOG_IO_S` — src/program/generate.cpp:9831
- `STRATA_WATCHDOG_S` — src/program/generate.cpp:9827
- `STRATA_WDDM_BUDGET` — src/core/device.cu:188
- `STRATA_WMMA_BF16` — src/prefill/gemm.cu:643
- `STRATA_WMMA_GEMM` — src/prefill/gemm.cu:642
- `STRATA_WMMA_GEMM` — src/prefill/gemm.cu:744
