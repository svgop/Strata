# F-457 trace - fold `gdn_ab_multi_kernel` into a neighbour (solo decode window graph)

**VERDICT: STOP.** The premise is refuted by the capture: in the champion's solo window graph `gdn_ab_multi_kernel` is NOT on the
window's critical path.  It runs on the side mixer's branch beside the qkv projection (sm120 default `STRATA_SIDE_MIXER=2`,
`verify.cpp:1549-1555`), and its only in-window reader, `gdn_step_norm_multi_kernel`, starts a median 16.90 us after it ends
(min 5.82; p10 12.54).  `gdn_ab` is the last of the step's producers to finish in **0 of 24,804** instances.  Removing its launch
therefore removes 0 us from the window.  The one bitwise, block-local fold available (into the consumer: step block `head` reads
only `gate[t*h_v+head]`/`beta[t*h_v+head]`) would move the a/b body in series AFTER the conv + gate projection - it ADDS its body's
time to the path and removes nothing: expected negative.  Folding into the producer of `xm` needs a grid-wide barrier (each a/b
warp's dot reads all N columns of `xm`): the brief's own STOP criterion.  Re-open clauses at the end.

Capture: `/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles/f453g-off-2026-11-25/anatomy.sqlite` (node-level, solo lane, m1).
Scripts (this directory, read-only): `f457-neighbours.py` (neighbours), `f457-slack.py` (slack to the consumer).  The `sqlite3` CLI is
not installed on this box; python3 `sqlite3` with `?mode=ro` is used.

## (a) Timeline neighbours inside the window graphs (measured)

1,448,686 kernel rows; window graphs = graphIds 4, 12, 20, 28, 36, 44 (1,377-1,879 kernels per launch, 25.4-34.8 `gdn_ab` per launch;
the other >= 1,500-kernel graphIds, 60-135, are 5-374-kernel graphs, 60 carries 144 `gdn_ab` in all).  3,497 window launches.
**`gdn_ab_multi_kernel` instances inside window graphs: 24,804** (matches the brief's ~24,800).  Grid 48 x 64 threads = 2 warps x 96 rows.

| slot | kernel | grid x block | median us | p10 | p90 | instances |
|---|---|---:|---:|---:|---:|---:|
| target | `gdn_ab_multi_kernel` | 48 x 64 | 9.31 (banked, F-453 chain 4B; mean 9.06) | 5.47 | 12.13 | 24,804 |
| BEFORE, GPU start order, all streams | `gr_up_v3_kernel` | 160 x 256 | 9.38 | 7.33 | 12.42 | 24,804 / 24,804 |
| AFTER, GPU start order, all streams | `native_quantize_q8_1_kernel` | 10..80 x 256 | 0.83-0.86 | 0.80 | 1.18 | 24,804 / 24,804 |
| AFTER, next reader of its output | `gdn_step_norm_multi_kernel` | 48 x 128 | 5.70 | 4.61 | 8.48 | 24,804 / 24,804 |

Gaps (all 24,804 instances): `gr_up_v3` END -> `gdn_ab` START median 0.19 us (p10 0.16, p90 0.19).
**`gdn_ab` END -> `gdn_step_norm` START: median 16.90 us, mean 16.83, p10 12.54, p1 9.66, min 5.82, p90 20.61.**
`gdn_conv_l2` END -> `gdn_step_norm` START: median 2.14 us (p10 0.19, p90 5.12).  Conv END <= `gdn_ab` END: 0 of 24,804.

One layer as the GPU ran it (window graph 28, `gdn_ab` ordinal 17; 14 launches superposed; us relative to `gdn_ab` START):

```
-10.3 ->  -0.1   10.2   160x256   gr_up_v3_kernel                 2nd half of the stream-split hyper-connection read: xm = mixed_ ready
  0.0 ->   9.3    9.2    48x64    gdn_ab_multi_kernel   TARGET    on the side branch
+0.15 ->  +1.0    0.83   40x256    native_quantize_q8_1_kernel     quant_in(xm): starts 0.13-0.29 us after gdn_ab, same input
+1.15 ->  +2.1    0.9    10x256    native_q8_1_interleave_kernel
+2.2  -> +20.2   18.0  1280x128    native_mmvq_il_kernel<IlQ6K>    qkv projection (grid 1280; block mapping not read)
+11.6  -> +24.1   12.5   768x128    native_mmvq_il_kernel<IlQ4K>    gate projection z (grid 768), on its own branch
+20.4  -> +23.3   2.8    80x128    gdn_conv_l2_multi_kernel        the qkv's conv (grid 80 x 128), on cs
+24.2  -> +30.6   6.4    48x128    gdn_step_norm_multi_kernel     starts 0.1 us after the LAST producer ends (z END +24.1)
```

Reading (measured): for ~15 us of every layer the branch that holds `gdn_ab` has nothing to run; the step is gated by the projection +
conv chain.  The path through this part of the layer is `gr_up_v3` -> quantize -> interleave -> mmvq_il -> conv -> `gdn_step_norm`.
Caveat on the capture: the reported `streamId` of graph nodes is not the capture stream (the code launches the conv on `cs`, `:1604`,
and the step on `cs`, `:1644`; the capture reports them on ids 64 and 62, the step beside `gdn_ab`), so the dependencies in (b) come
from the code and only the timestamps from the capture.

## (b) Launch sites and data dependencies (code, `src/core/verify.cpp`; static)

Three launch sites; the champion's solo graph uses the first (on compute capability 12.0 `g_side_mixer()` returns 2 unless
`STRATA_SM120_PERF=0` or `STRATA_SIDE_MIXER` is set, `verify.cpp:187-196`; `br` is false whenever the side mixer is on, `:1266`):

| site | condition | stream | line |
|---|---|---|---|
| active | `sidem` = side mixer on and not a batch window | `side_`, forked from `cs` by `branch(mfork_, cs, side_)` | 1549-1555 |
| - | `br` = `STRATA_DF_BRANCH` and no side mixer | `df_side_[0]` | 1573-1577 |
| - | neither (batch windows, or the mixer off) | `cs`, in line between the conv and the gate projection | 1611-1614 |
| - | `fuse_ab` = `STRATA_FUSE_CONVAB`, HIP only (`fuse_convab_on()` returns false on CUDA, `:376-383`) | - | 1598-1602 |

Kernel (`verify_kernels.cu:187-194`, body `:115-185`, launcher `:1174-1193`): one warp per row, 2 rows (a, b) per block,
`(2*h_v+1)/2` blocks = 48 => h_v = 48; each lane runs 10 passes (`j = lane + 32 jj`, all < n/8 = 320, fast path when n == 2560) of
8 FMAs in a dependent chain per token, then a 32-lane `__shfl_xor_sync` reduction; lane 0 writes.

- READS: `xm` (the layer input, n tokens x N f32 - `xm = mixed_ + tb*N`, `:1485`, produced by the stream-split hyper-connection
  read, `gr_down_v3_kernel` / `gr_up_v3_kernel`, `fused_gr.cu:2784`, on `cs` before the fork; `gr_up_v3_kernel` ends 0.19 us
  before `gdn_ab` starts, 24,804 / 24,804); `ssm_alpha.weight` and `ssm_beta.weight` (bf16, 48 x N each, 480 KB),
  `ssm_dt.bias` and `ssm_a` (f32[48]) - weights, read-only.
- WRITES: `gate[t*48 + r]` = softplus(a_r + dt_r) * A_r and `beta[t*48 + r]` = sigmoid(b_r), f32, n x 48 each.  No other output.
- CONSUMERS: `gdn_step_norm_multi_kernel` (the recurrence; `cs` waits on `mjoin_` at `:1622`; block `head` reads only
  `gate[t*h_v + head]`, `beta[t*h_v + head]` - `verify_kernels.cu:389,408,432`) and, in the commit path, `gdn_step_commit_body`.
  In the capture the next `gdn_step_norm_multi_kernel` of the same launch follows in 24,804 / 24,804 instances.

## (c) The neighbour to fold with

- Consumer `gdn_step_norm_multi_kernel`: the only fold that is bitwise without a new layout - block `head` would compute its own two
  rows with warps 0 and 1 (same FMA chain, same shuffle reduction), no barrier.  But the fused kernel can only START when the conv
  and the gate projection have ended (+24.2 in the layer above); the a/b body then runs in series.  Per layer the path grows by the
  body's marginal time and loses nothing, because the launch it removes was not on the path.  This is the serialization that the
  in-tree conv + a/b fold (`gdn_conv_ab_kernel`, 143-amdk, `verify_kernels.cu:196-212`) already showed on gfx1201: -0.3%
  (`verify.cpp:375`: "the a/b dot chain is the longer one").
- Producer of `xm` (`gr_*_v3_kernel`): every a/b warp reads all N columns of `xm`, so the fold needs all of `xm` written before any
  a/b block starts: a grid-wide barrier.  STOP criterion.
- The qkv projection / gate projection: same input, but their weights are native-quantized (`IlQ6K`, `IlQ4K`) - folding means one
  matvec over concatenated rows in one layout: a new data layout.  STOP criterion.  (A co-launch of two unrelated bodies in one grid,
  the `gdn_conv_ab_kernel` form, is bitwise but moves `gdn_ab` behind the qkv projection: same loss as above.)

## (d) Price (the launch cost removed per layer x 36)

The brief's 5-9 us x 36 = 0.18-0.32 ms per window = +1.8-3.2% M1 holds only if the launch sits on the window's path.  Measured: it does
not (0 / 24,804 instances last to finish).  **Removed from the path: 0 us per layer = 0.0% M1** (node-level measurement; no end-to-end
run).  Cost of the one feasible fold, if built: the a/b body's marginal time per layer x 36 / 10.2 ms.  Marginal time UNMEASURED
(standalone 9.06 us mean includes the launch floor the record puts at p10 = 5.5 us): 2 us -> -0.07 ms = -0.7% M1; 3.5 us -> -0.13 ms
= -1.2%; the whole 9 us -> -0.32 ms = -3.1%.  Every case is negative.

## (e) Re-open clauses

1. A lane where the side mixer is off.  Batch windows disable it (`side_proj`/`sidem` carry `!batch_rec_`, `verify.cpp:1232,1239`), so
   there `gdn_ab` is launched in line at `:1612` between the conv and the gate projection and its launch IS on the path.  That is the
   concurrent lane's window, not this one.  UNCONFIRMED static analysis: no node-level capture of the batch window was read.
2. If the projection branch through quantize -> mmvq_il -> conv -> z gets more than 17 us per layer faster, `gdn_ab` (9 us) becomes the
   last producer of the step and the consumer fold (or a faster a/b body) turns into a lever.  Falsifier: re-run `f457-slack.py` on a
   new capture; the lever exists when "last producers to finish" leaves 0.
