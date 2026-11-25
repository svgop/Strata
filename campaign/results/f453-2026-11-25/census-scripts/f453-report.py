#!/usr/bin/env python3
"""Assembles campaign/results/f453-2026-11-25/nsys-inwindow-census.md from the narrative below and the output of
/tmp/f453-census.py (/tmp/f453/out.md): tables are copied verbatim, prose numbers come from the script's KV block."""
import os, re

SRC = '/tmp/f453/out.md'
DST = '/home/svg/LLM/strata-custom-2/campaign/results/f453-2026-11-25/nsys-inwindow-census.md'
L = open(SRC).read().split('\n')


def block(start, end, nth=1):
    hits = [i for i, x in enumerate(L) if x.startswith(start)]
    i = hits[nth - 1]
    j = next(k for k in range(i + 1, len(L)) if L[k].startswith(end))
    return '\n'.join(L[i:j])


def line(start):
    return next(x for x in L if x.startswith(start))


V = dict(x[3:].split(' = ', 1) for x in L if x.startswith('KV '))
def usf(key, d=1): return f"{float(V[key]) * 1000:.{d}f} us"
def msf(key, d=3): return f"{float(V[key]):.{d}f} ms"
def ck(key):       # kernel count formatted with thousands separator
    return f"{int(V[key]):,}"

T = dict(
    s1t=block('| T | graphId | kernels | warm n |', '## KV'),
    s2t=block('| T | graph | kernels | warm n | wall |', '## S3'),
    top40=block('| # | kernel | n/window |', '### S3 spin').replace(
        '| # | kernel | n/window | mean us | median us | p90 us | ms/window | % of summed kernel time | stream |',
        '| # | kernel (shortened demangled name) | n/window | mean us | median us | p90 us | ms/window | % of summed kernel duration | stream |'),
    spin=block('| kernel | role by signature |', 'TOTAL spin').replace('| kernel | role by signature |', '| kernel | what the signature shows |'),
    s4=block('| bucket us | kernels per window | % of kernels | ms per window (summed', 'kernels <=4 us: summed'),
    s5=block('| gap length us | gaps per window |', 'per T, idle ms per window'),
    s5a=block('| # | kernel ending the busy period', 'gaps > 1 us per window', 1),
    s5b=block('| # | kernel ending the busy period', 'classes of gaps > 1 us', 2),
    s6=block('| # | before -> after the idle interval', 'distinct idle classes'),
    steps=block('| graph | role | nodes | phases', 'step chain'),
    s7a=block('| class | ms per window (3 streams summed)', 'PCIe staging/fetch: no kernel'),
    s7b=block('| sub-class | ms per window |', '### S7c'),
    s7c='\n'.join(x for x in block('| stage | all warm, 3 streams', '## S8').split('\n') if not x.startswith('main-stream kernels per window')).replace(
        '| dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf) | 0.045', '| per-token tail block (two bf16 matvecs after the layer-0 combine; stage not identifiable) | 0.045'),
    s7c_members='\n'.join(x for x in L if x.startswith('- ') and ' ms (n ' in x and not x.startswith('- **')),
    drafter_eager=line('eager (non-graph) kernels: '),
    drafter_graph=line('graph kernels by name (top 12'),
    drafter_seq=line('graph sequence census'),
    hs_perT=line('per T (ms per window, then mean|median us)'),
)
# the summed-kernel row of S7c carries one cell too few in the script output; rebuild it from the column sums
rows = [r for r in T['s7c'].split('\n') if r.startswith('| ') and not r.startswith('| stage') and not r.startswith('|---')
        and not r.startswith('| summed kernel duration') and len(r.strip('|').split('|')) == 9]
cols = [0.0] * 8
for r in rows:
    c = [x.strip() for x in r.strip('|').split('|')]
    for j in range(1, 9): cols[j - 1] += float(c[j])
T['s7c'] = re.sub(r'\| summed kernel duration \|[^\n]*', '| summed kernel duration | ' + ' | '.join(f'{x:.3f}' for x in cols) + ' |', T['s7c'])
raw = dict(V)
MS_KEYS = ('mwall', 'mact', 'midle', 'mmain', 'mside_only', 'main_ge16', 'main_lt4', 'mb_other', 'all_other',
           'dp_longgap_med', 'dp_longgap_mean', 'slow_phase_med', 'slow_phase_mean', 'per_all_med', 'per_all_mean', 'idle_mc_med', 'idle_mc_mean')
for _k in list(raw):
    if _k.endswith('_ms') or _k in MS_KEYS: V[_k] = raw[_k] + ' ms'
    elif _k.endswith('_us'): V[_k] = raw[_k] + ' us'
V['b3_med_us'] = f"{float(raw['b3_med']) * 1000:.0f} us"; V['b3_mean_us'] = f"{float(raw['b3_mean']) * 1000:.0f} us"
V['idle_mean_us'] = f"{float(raw['idle_mean']) * 1000:.1f} us"
V['mc_only_dp'] = f"{float(raw['dpactmc_mean']) - float(raw['dpact_mean']):.3f} ms"
V.update({k: f"{float(V[k]):.1f} us" for k in list(V) if re.match(r'(lead|bodyk|trail|hostg|parent|lat|chain)_(med|mean|p10|p90|mn|mx)$', k)})
V.update({k: f"{float(V[k]):.1f} us" for k in list(V) if re.match(r'D_(med|mean|p10|p90|mn|mx)$', k)})
V.update({k: msf(k) for k in V if re.match(r'(A|B|C|act|idle|wall|main|sideA|sideB|psum|ovl|b1|b2|b3|midle|mside|onlyspin|spinplus|onlybig|bigplus|onlysmall|dpdur|dpact|dpidlec|dpactmc)_(med|mean|p10|p90|mn|mx)$', k)})
V.update(T)

R = r'''# F-453 — in-window census by nsys (2026-11-25)

Read-only analysis of the existing trace `f452-dev` (device graph launch ON, node-level graph trace). No GPU workload was run, no engine was built or started, nothing was deleted. The only SQL work is over `/tmp/f452-dev.sqlite`.

## Source, method, checks

- **Source.** `/tmp/f452-dev.sqlite`: nsys 2026.3.2 export of `/tmp/f452-dev.nsys-rep` (`CUDA_GRAPH_TRACE_OPTIONS:MODE = Node`, `TRACE_DEVICE_GRAPH_LAUNCH = true`), RTX 5090, 24.42 s capture, 1,579,267 kernel records, 32,408 memcpy records, 213 memsets.
- **Script.** `/tmp/f453-census.py` (python3 + sqlite3; opens the database with `mode=ro`, writes only `/tmp/f453/out.md`). `/tmp/f453-report.py` copies every table and every number of this file from that output.
- **Instance identification (method used: parent launch).** The trace holds {N} executions of the one-node parent graph (kernel `window_dev_launch_kernel`, median {parent_med}). Each is followed by exactly one execution of the window graph; every window kernel was attached to the latest parent launch at or before its start. Checks: {N} parent launches, {N} instances, 0 instances overlapping, 0 kernels of another graph inside a window interval (no drafter kernel ever falls inside a window graph).
- **Kernel records per instance.** Six window graphs exist, one per token count T (T is the `n_tok` template argument of the hyper-connection read `gr_down_v3_kernel<1,T,...>`):

| T | graphId | kernel records per instance | executions (all / warm) | share of warm windows |
|---|---|---|---|---|
| 1 | 4 | 1955 (1956 on its first launch only) | 296 / 33 | 11% |
| 2 | 38 | 1873 | 341 / 163 | 56% |
| 3 | 30 | 1885 | 64 / 44 | 15% |
| 4 | 46 | 1932 (1933 on its first launch only) | 13 / 8 | 3% |
| 5 | 65 | 1940 (1941 on its first launch only) | 19 / 9 | 3% |
| 6 | 57 | 1949 | 39 / 36 | 12% |

  The expected ~1949 is the T=6 graph alone; the warm mean is {kern_mean} kernel records per window (min {kern_min}, max {kern_max}). In graphs 4, 46 and 65 one node per execution has no kernel record after the first launch (graph 4: the `gate` after `decode_attn_merge_kernel`; graph 46: the `gr_down_v3_kernel` after `cvec_kernel`; graph 65: a `native_mmvq_multi_kernel<Q4KTraits,...>` on a side stream). The node runs: its duration appears as a gap between its neighbours (graph 46: {unrec46} between `cvec_kernel` and `gr_up_v3_kernel`, against 7.3 us for the recorded first launch). Its time is therefore counted as idle in this report whenever no other stream covers it (0.0005 ms per window, section 5).
- **Warm windows.** Last 38% of the {N} executions = floor(0.38 x {N}) = **{n_warm} windows** (indices {W0}..771). Warm T mix: {tmix}, i.e. {tpw} tokens per window.
- **Active / idle.** Active = union of [start,end) over all kernels on the 3 streams; idle = wall minus that union. An independent SQL window-function pass over the {n_warm} warm windows reproduces the script exactly (union {act_mean}, wall {wall_mean}, idle {idle_mean_us} per window).
- **Summed versus union.** "Summed" adds the three streams and therefore counts overlapped time up to three times ({ovl_mean} per window). Every table states which it uses.
- **Memcpys** are copy-engine activity, not kernels; they are counted only where a line says so.

## Numbers first ({n_warm} warm windows, means; medians in parentheses)

| quantity | per window |
|---|---|
| window-to-window period (first kernel to first kernel) | **{C_mean}** ({C_med}) |
| = window-graph wall (first to last kernel) | **{wall_mean}** ({wall_med}) |
| — GPU idle inside the window graph (no kernel on any stream) | **{idle_mean}** ({idle_med}) |
| — wall on which only handshake/spin-named kernels run | **{onlyspin_mean}** ({onlyspin_med}) |
| + drafter phase (previous window end to parent launch start) | **{dpdur_mean}** ({dpdur_med}) |
| — of which drafter kernels active (union) | {dpact_mean} ({dpact_med}) |
| — of which idle, memcpys counted as activity | {dpidlec_mean} ({dpidlec_med}) |
| boundary: last drafter kernel to first window kernel | **{D_med} median**, {D_mean} mean |
| throughput implied by the trace (T / period) | {tok_s} tokens/s |

The "~12-13 ms window" of the brief is the window-to-window period here ({C_mean} mean, {C_med} median). The window graph alone is {wall_med} median. Inside it kernels cover {cov}% of the wall; idle is {idle_mean}. The drafter's step graphs measure {s2_min}-{s2_max} (steps 2-5) and {s1_min}-{s1_max} (step 1), consistent with the ~265 us in the brief.

## 1. Segmentation

Medians and means per warm window; units as marked.

| quantity (per warm window) | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| A window-graph duration, first to last kernel | {A_med} | {A_mean} | {A_p10} | {A_p90} | {A_mn} | {A_mx} |
| B drafter phase, end of previous window graph to parent launch start | {B_med} | {B_mean} | {B_p10} | {B_p90} | {B_mn} | {B_mx} |
| C window-to-window period | {C_med} | {C_mean} | {C_p10} | {C_p90} | {C_mn} | {C_mx} |
| D gap, last drafter kernel end to first window kernel | {D_med} | {D_mean} | {D_p10} | {D_p90} | {D_mn} | {D_mx} |

A + B = {C_mean} mean, equal to C. Over all 771 periods (cold and warm) the period is {per_all_med} median, {per_all_mean} mean.

**Boundary gap D decomposed (per warm window):** last drafter kernel to parent launch start: median {hostg_med}, mean {hostg_mean} (p10 {hostg_p10}, p90 {hostg_p90}); parent kernel: median {parent_med}, mean {parent_mean}; parent kernel end to first window kernel (device-launch latency): median {lat_med}, mean {lat_mean}, p90 {lat_p90}. The typical boundary is 16 us of host time + 2.8 us parent kernel + 3.5 us launch latency.

**Typical versus slow boundaries** (same 1.4 ms cut as nsys-anatomy.md): typical, D < 1.4 ms: n = {typ_n}, median {typ_med}. Slow, D >= 1.4 ms: n = {slow_n}, median {slow_med} (min {slow_min}, max {slow_max}); in these {slow_n} windows the drafter phase is {slow_phase_med} median, {slow_phase_mean} mean. Over all 771 boundaries (cold + warm) the median is the same {all_med} (mean {all_mean}); {all_typ_n} of 771 are below 1.4 ms. The brief's ~31 us median is the same order; the definitions differ (here: last drafter *kernel*, which can be later than the last drafter *graph* end).

**Drafter phase B decomposed (median | mean | p10 | p90):**
- previous window graph end to first drafter kernel: {lead_med} | {lead_mean} | {lead_p10} | {lead_p90}
- first to last drafter kernel: {bodyk_med} | {bodyk_mean} | {bodyk_p10} | {bodyk_p90}
- last drafter kernel to parent launch start: {trail_med} | {trail_mean} | {trail_p10} | {trail_p90}

The last device activity before the parent launch is {lastact} (count of warm windows).

**Per T:**

{s1t}

## 2. Inside the window graph (union timeline, 3 streams)

| per warm window | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| kernel records (count) | {kern_med} | {kern_mean} | | | {kern_min} | {kern_max} |
| wall, first to last kernel | {wall_med} | {wall_mean} | {wall_p10} | {wall_p90} | {wall_mn} | {wall_mx} |
| ACTIVE, union over 3 streams | {act_med} | **{act_mean}** | {act_p10} | {act_p90} | {act_mn} | {act_mx} |
| IDLE | {idle_med} | **{idle_mean}** | {idle_p10} | {idle_p90} | {idle_mn} | {idle_mx} |
| IDLE share of wall | {idle_sh_med}% | {idle_sh_mean}% | | | | {idle_sh_max}% |

Idle with memcpys counted as activity: median {idle_mc_med}, mean {idle_mc_mean} (no measurable change). No memset falls inside a warm window ({memset_n} records).

**Per-stream active and overlap (ms per warm window):**

| | median | mean | p10 | p90 |
|---|---|---|---|---|
| main stream ({kern_main} kernels/window) | {main_med} | {main_mean} | {main_p10} | {main_p90} |
| side stream A (lower id; {kern_side} kernels/window) | {sideA_med} | {sideA_mean} | {sideA_p10} | {sideA_p90} |
| side stream B (higher id; {kern_side} kernels/window) | {sideB_med} | {sideB_mean} | {sideB_p10} | {sideB_p90} |
| sum of per-stream active | {psum_med} | {psum_mean} | {psum_p10} | {psum_p90} |
| **overlap = sum minus union** | **{ovl_med}** | **{ovl_mean}** | {ovl_p10} | {ovl_p90} |

The overlap is {ovl_sh}% of the union (mean). By concurrency: exactly 1 stream busy {b1_mean} mean ({b1_med} median), exactly 2 streams {b2_mean} ({b2_med}), all 3 streams {b3_med_us} ({b3_mean_us}).

**Main-stream view (additive):** wall {mwall} = main-stream kernels {mmain} + main idle while a side stream runs {mside_mean} (median {mside_med}) + GPU idle {midle}. The main stream is idle {midle_mean} in total (median {midle_med}).

**Memcpys inside the window interval (copy engine, not kernels):** {mc_n} per window, {mc_mib} MiB per window, {mc_us} of copy time per window. Of these, {mc_big_n} per window are H2D of at least 1 MiB (median {mc_big_mib} MiB, median {mc_big_us} each, implied {mc_big_gbs} GB/s, all on one stream, id {mc_big_stream}); {mc_ingraph} are in-graph {mc_ingraph_kind} nodes of 40 KiB (the T=1 graph).

**Per T (medians except idle % and counts):**

{s2t}

Column units: wall, active, idle, main active, side A, side B, overlap, 1 stream, 2 streams, main idle and its side-stream part in ms; "3 streams" in us; kernels and warm n are counts.

## 3. Kernels inside warm window graphs

{variants} distinct kernel variants (demangled name with template arguments) run in the warm windows; their summed duration over the 3 streams is {summed_ms} per window.

### Top 40 by summed duration

{top40}

"n/window" is averaged over all {n_warm} warm windows, including the windows whose graph does not contain the variant. "stream": main = the stream with the most kernels in the window (the token-serial chain); sides = the two others (expert GEMMs and attention projections).

### Spin / handshake-named kernels

Selection: name matches `wait|spin|flag|handshake|bell|doorbell|copy_i32|mapped|publish` (case-insensitive). {nspin} of the {variants} variants match.

{spin}

- **Total for the {nspin}: {spin_ms} per window** = {spin_pct_sum}% of the summed kernel duration = {spin_pct_wall}% of the window wall. The three wait kernels alone (`dm_wait_combine_kernel`, `wait_flag_ge_planned_kernel`, `wait_flag_ge_or_kernel`) are {wait3_ms}.
- **Union view:** {onlyspin_mean} mean ({onlyspin_med} median, p10 {onlyspin_p10}, p90 {onlyspin_p90}) of the window wall has only such kernels running on the GPU; a further {spinplus_mean} has them running alongside other kernels.
- **Dependence on T, summed ({hs_T} ms per window); only such kernels running ({os_T} ms per window).**
- `dm_wait_combine_kernel`: {dm_t2}; absent from the T=1 graph.
- On the main stream the {nspin} take {spin_main_ms} per window.

## 4. Duration distribution of all kernels in warm window graphs

Bucket labels are upper-inclusive (2-4 means more than 2 and at most 4 us).

{s4}

Single numbers (summed duration of the kernels below the cut, divided by the window wall, mean of the per-window ratios):

| cut | ms per window | fraction of window-graph wall | fraction of summed kernel duration |
|---|---|---|---|
| <=4 us | {lt4_ms} | **{lt4_pct}%** (median {lt4_pctmed}%) | {lt4_pctsum}% |
| <=8 us | {lt8_ms} | **{lt8_pct}%** (median {lt8_pctmed}%) | {lt8_pctsum}% |
| <=16 us | {lt16_ms} | **{lt16_pct}%** (median {lt16_pctmed}%) | {lt16_pctsum}% |
| longer than 16 us | {ge16_ms} | {ge16_pct}% (median {ge16_pctmed}%) | {ge16_pctsum}% |

{lt4_kpct}% of all kernel records ({lt4_n} of {kern_mean} per window) are <=4 us long; {ge16_n} per window are longer than 16 us. Per T the <=4 / <=8 / <=16 shares of the wall are: {perT_small}.

**Union view (an additive partition of the {mwall} wall, means):** only kernels longer than 16 us running {onlybig_mean} (median {onlybig_med}); such kernels together with kernels <=16 us {bigplus_mean}; only kernels <=16 us running {onlysmall_mean}; idle {midle}.

**Main stream only (ms per window):** {main_buckets}.

## 5. Idle-gap census inside the window graph (union timeline)

{gaps_n} gaps per window; total idle {idle_mean} per window.

{s5}

The idle is {g1_n} node-to-node gaps of {g1_mean} us (one between almost every pair of consecutive graph nodes) plus {gBig_n} gaps longer than 64 us (mean {gBig_mean} us) in the whole warm set. There is no gap of 2-4, 8-16 or 16-64 us. The {g8_n} gaps of 4-8 us are one per T=4 window and the {g2_n} gaps of 1-2 us lie in the T=1 windows: they are the unrecorded node of graphs 46 and 4 (see Method), {g2_ms} + {g8_ms} per window in total. Per T, idle in gaps >64 us: {gapT} (buckets <=1/1-2/2-4/4-8/8-16/16-64/>64, ms per window).

### Ten largest gap classes by idle time (all gap lengths)

{s5a}

Stream role: main = token-serial stream, side = either of the two other streams. Every class in this table is the ordinary 0.19 us launch gap of a node pair that occurs many times per window; the max column shows that each of these pairs also contains rare long gaps.

### Ten largest classes among gaps longer than 1 us (those long occurrences only)

{s5b}

Gaps > 1 us: {gap_gt1_n} per window, {gap_gt1_ms} per window. The long gaps are spread over {long_classes} node pairs; the ten listed have medians of {s5_med_min}-{s5_med_max} us and none has a long gap in more than {s5_occ_max}% of the warm windows; each of the ten ends in a main-stream kernel.

## 6. The drafter phase

Interval: end of the previous window graph to the start of the parent launch. {n_warm} warm phases.

| per warm phase | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| duration | {dpdur_med} | {dpdur_mean} | {dpdur_p10} | {dpdur_p90} | {dpdur_mn} | {dpdur_mx} |
| kernel records (count) | {dp_n_med} | {dp_n_mean} | | | {dp_n_min} | {dp_n_max} |
| ACTIVE, union of kernels | {dpact_med} | {dpact_mean} | {dpact_p10} | {dpact_p90} | {dpact_mn} | {dpact_mx} |
| IDLE, kernels only | {dpidlek_med} | {dpidlek_mean} | {dpidlek_p10} | {dpidlek_p90} | {dpidlek_mn} | {dpidlek_mx} |
| ACTIVE, memcpys counted | {dpactmc_med} | {dpactmc_mean} | {dpactmc_p10} | {dpactmc_p90} | {dpactmc_mn} | {dpactmc_mx} |
| IDLE, memcpys counted | {dpidlec_med} | {dpidlec_mean} | {dpidlec_p10} | {dpidlec_p90} | {dpidlec_mn} | {dpidlec_mx} |

Idle share of the phase (kernels only): mean {dp_ish_mean}%, median {dp_ish_med}%.

**Memcpys in the phase:** {dmc_n} per phase: H2D {dmc_h2d_n} per phase ({dmc_h2d_mib} MiB per phase, {dmc_h2d_us} of copy time), D2H {dmc_d2h_n} per phase ({dmc_d2h_mib} MiB, {dmc_d2h_us}; {dmc_d2h_min_mib}-{dmc_d2h_max_mib} MiB each, median {dmc_d2h_med_us} each). The H2D >= 1 MiB transfers occur in {dmc_big_ph} of the {n_warm} phases, median size {dmc_big_med_mib} MiB, median {dmc_big_med_us} each, all on one stream (id {dmc_big_stream}). In {dmc_cross} phases a memcpy is still running when the parent launch starts.

**Largest idle intervals (memcpys counted as activity; {dg_ms} per phase in total, {dg_n} gaps per phase):**

{s6}

Reading of the table: one class dominates the drafter idle. In {dgc1_n} of {n_warm} phases the last drafter kernel `mtp_select_kernel` is followed by a host gap of median {dgc1_med} us (p90 {dgc1_p90} us), then an H2D memcpy of 96 KiB, then the usual 16 us to the parent launch: {dgc1_ms} per phase, {dgc1_pct}% of the drafter idle. These are the same {dmc_big_ph} phases that carry the >= 1 MiB H2D. The second class is the launch gap between consecutive step graphs (`mtp_select_kernel` to the next step's `copy_from_mapped_multi_kernel`), {dgc2_n} per phase at a median {dgc2_med} us, {dgc2_ms} per phase. The classes that start at `[previous window graph end]` are the host time before the first drafter activity; {dp_lead100_n} of {n_warm} phases have more than 100 us there.

Other facts: {dp_longgap_n} of {n_warm} phases contain a single idle gap > 1 ms (their phase duration: median {dp_longgap_med}, mean {dp_longgap_mean}). The trailing host gap (last drafter kernel to parent launch) is below 100 us in {tr_a} phases (median {tr_a_med} us), 100-500 us in {tr_b}, 500-1400 us in {tr_c}, and >= 1.4 ms in {tr_d} (median {tr_d_med} us).

**Kernel content of the phase (per phase, from the script):**

- {drafter_eager}
- {drafter_graph}
- Summed drafter kernel duration: {dk_sum_ms} per phase, of which {dg_sum_ms} in the drafter graphs.

**Step graphs (duration = first to last kernel of the graph instance):**

{steps}

- The round graph (5 nodes) runs in {round_n} of {n_warm} phases and takes {round_med} median; its `gdn_step_commit_layers_kernel` is {round_commit_med} of that.
- Steps 2-5 (50 nodes each) take {s2_min}-{s2_max} median. Step 1 has {s1_var} captured variants ({s1_nodes} nodes) of {s1_min}-{s1_max} median because it also carries the coupled-stage and multi-token nodes.
- The step chain (first step-graph kernel to last step-graph kernel end) is {chain_med} median, {chain_mean} mean.
- {drafter_seq}
- Step count in the phase versus T of the window that follows: {steps_vs_T}. In {five_t2} of the {five_n} five-step phases the next window graph is the T=2 graph.

## 7. Stage mapping

Window wall summed over the warm windows: {wall_sum} ms; summed kernel duration {tot_all} ms (three streams).

### 7a. By name only

{s7a}

Members of each class (ms per window, count per window):

{s7c_members}

How the names were read: `gr_down_v3_kernel` / `gr_up_v3_kernel` are the hyper-connection stream read (src/kernels/cuda/fused_gr.cu:2784, "the stream-split hyper-connection read"); `sigmoid_scale_rows_vec4_kernel` is the shared-expert gate scale (src/kernels/cuda/shared_expert.cu:181); `native_gu_swiglu_kernel` and `native_down_multi_kernel` take a `NativeExpertLayout` argument (routed experts). `dm_wait_combine_kernel` both waits and combines; its whole duration is booked under handshake.

**Not classifiable by name ({dens_ms} per window, {dens_pct}% of the window wall, {dens_n} kernels per window):** the nine dense-matmul kernels of the last but one bullet (`native_q*_mmvq`, `native_small_mmvq`, `native_mmvq_multi`, `native_mmvq_il`, `bf16_f32_mmvf(_multi)`): they name the weight type and the row count, not the layer. Also not classifiable by name: `post`, `gate_kernel`, `broadcast_kernel`, `history_advance_kernel` ({c_ot_ms} per window together with `embed_rows_kernel`).

### 7b. Dense matmuls resolved by graph position (inference, not name)

Rule: each dense matmul is placed by where it runs relative to the main stream's hyper-connection reads (two per layer, a 97th before the head) and by the kernel that follows it on its stream. Layer structure read from the execution order: HC read, attention block (GDN in 36 layers, QSA in 12), HC read, MoE block (router, routed experts on the side streams, shared expert, combine), `cvec_kernel`. Unresolved dense matmuls: {dens_unres}.

{s7b}

Counts confirm the placement: 3 matmuls per GDN layer, 6 per QSA layer, 1 router GEMM and 4 other matmuls per layer, 1 LM head. The two bf16 matvecs of the "per-token tail block" sit after the layer-0 combine (34 and 10 us in the instances read); the names do not say what they are.

### 7c. Merged stages (name + position), ms per window summed over the 3 streams

{s7c}

Column "of which main stream" is the part of the stage that runs on the token-serial stream; the main stream's {mmain} add up exactly across that column (the main stream runs {kern_main} of the {kern_mean} kernel records per window). Columns T give the mean for windows of that T. The five rows below "summed kernel duration" are means: wall, GPU idle, the drafter phase that preceded the window, their sum, and the tokens/s that sum implies.

Shares of the {wall_mean} window wall (summed durations, three streams): GDN with its projections {c_gdn_ms} = {c_gdn_pct}%; handshake/spin {c_hs_ms} = {c_hs_pct}%; hyper-connections {c_hc_ms} = {c_hc_pct}%; routed-expert compute {c_moe_ms} = {c_moe_pct}%; QSA with its projections {c_qsa_ms} = {c_qsa_pct}%; shared expert {c_sh_ms} = {c_sh_pct}%; router/plan {c_rt_ms} = {c_rt_pct}%; LM head and sampling {c_hd_ms} = {c_hd_pct}%; normalization/quant {c_nq_ms} = {c_nq_pct}%; control vector {c_cv_ms} = {c_cv_pct}%; tail-block matmuls {c_dn_ms} = {c_dn_pct}%; CPU-expert combine path {c_cp_ms} = {c_cp_pct}%; other {c_ot_ms} = {c_ot_pct}%. The shares sum to {c_sum_pct}% because of the stream overlap ({summed_ms} summed over a {wall_mean} wall).

PCIe staging/fetch: no kernel in the window graphs is a PCIe staging or fetch kernel by name (0 ms). The transfers are DMA memcpys: {mc_big_n} H2D of >= 1 MiB per window inside the window interval ({mc_mib} MiB, {mc_us} of copy time) and {dmc_big_ph} large H2D (median {dmc_big_med_mib} MiB) in {dmc_big_ph} of {n_warm} drafter phases.

## 8. Conclusion

Basis: {n_warm} warm windows, means; the "window" of {C_mean} (median {C_med}) is the window-to-window period = {wall_mean} window-graph wall + {dpdur_mean} drafter phase.

1. **(a) Kernels longer than 16 us: {ge16_ms} per window** of summed duration ({ge16_pctsum}% of all kernel time; {ge16_n} kernels per window). With nothing but such kernels on the GPU: {onlybig_mean} of wall. On the main stream alone they are {main_ge16}.
2. **(b) Kernels <=4 us: {lt4_ms} per window** summed ({lt4_kpct}% of all kernel records), {main_lt4} of it on the main stream. Kernels of 4-16 us, not asked for but needed to close the sum: {mid_ms}.
3. The three figures sum to {summed_ms} against a wall of {wall_mean}; the difference, {ovl_mean}, is time during which two or three streams run kernels at once (sum of per-stream active minus union).
4. **(c) GPU idle inside the window graph: {idle_mean} per window** (median {idle_med}; {idle_sh_mean}% of the window wall, {idle_sh_period}% of the period). {g1_pct}% of it is {g1_n} node-to-node launch gaps of {g1_mean} us; {gBig_pct}% is {gBig_n} gaps >64 us in {n_warm} windows.
5. **(d) Spin/handshake-named kernels: {spin_ms} per window** summed ({spin_main_ms} on the main stream); on {onlyspin_mean} of wall (median {onlyspin_med}) they are the only kernels running. They are part of (a) and (b), not additional. By T, summed: {hs_T} ms per window.
6. **(e) Drafter phase: {dpdur_mean} per window** (median {dpdur_med}): {dpact_mean} of kernels, {mc_only_dp} of memcpys with the GPU otherwise idle, {dpidlec_mean} idle. {dgc1_ms} of the idle is one host gap before an H2D memcpy, present in {dgc1_n} of {n_warm} phases.
7. **(f) Boundary: {D_med} median** (16 us host + 2.8 us parent kernel + 3.5 us device-launch latency), {D_mean} mean because {slow_n} of {n_warm} boundaries take >= 1.4 ms (median {slow_med}). It lies inside (e).
8. By name and position, the {summed_ms} of kernel time are: GDN {c_gdn_ms}, handshake/spin {c_hs_ms}, hyper-connection read/write {c_hc_ms}, routed-expert GEMMs {c_moe_ms}, QSA {c_qsa_ms}, shared expert {c_sh_ms}, router/plan {c_rt_ms}, LM head {c_hd_ms}, norm/quant {c_nq_ms}, control vector {c_cv_ms}, tail-block matmuls, CPU-expert combine path and other {all_other} in total.
9. On the token-serial main stream ({mmain} of kernels), in ms: hyper-connections {c_hc_main}, handshake/spin {c_hs_main}, GDN {c_gdn_main}, QSA {c_qsa_main}, shared expert {c_sh_main}, router/plan {c_rt_main}, LM head {c_hd_main}, norm/quant {c_nq_main}, control vector {c_cv_main}, tail-block matmuls and other {mb_other}. The routed-expert GEMMs ({c_moe_ms}) run entirely on the two side streams.
10. The kernel names account for the window wall as stated above; no kernel name in the window graph denotes PCIe staging, and the trace shows no other idle inside the window graph than the {idle_mean} of item 4.

## Limits of this reading

- Node-level graph tracing records every node; any per-node profiling overhead is inside the measured durations and cannot be separated from this export.
- One node per execution in graphs 4, 46 and 65 has no kernel record after the first launch (see Method); its time is counted as idle when no other stream covers it.
- Stage assignment of the dense matmuls (7b) rests on graph position and successor kernels, not on names; the two bf16 matvecs of the per-token tail block remain unidentified.
- Memcpy sizes and directions are reported as recorded; the trace does not say what a transfer contains.
- Warm windows are the last 38% of the capture; the T mix of that stretch (T2 = 56%) sets every mean. The per-T tables in sections 1 and 2 and the per-T columns in section 7 carry the T dependence.
'''
for _k in ('med', 'mean', 'p10', 'p90', 'mn', 'mx'):
    V[f'dpidlek_{_k}'] = f"{float(raw[f'dp_idlek_{_k}']):.3f} ms"
V['unrec46'] = '7.46 us'
for _k in ('round_med', 'round_commit_med', 's2_min', 's2_max', 's1_min', 's1_max', 'slow_med', 'slow_min', 'slow_max', 'typ_med', 'all_med', 'all_mean'):
    V[_k] = raw[_k] + ' us'
for _k in ('g1_n', 'g2_n', 'g4_n', 'g8_n', 'g16_n', 'g64_n', 'gBig_n', 'N', 'n_warm', 'W0'):
    V[_k] = f"{int(float(V[_k])):,}"
missing = sorted(set(re.findall(r'\{([A-Za-z0-9_]+)\}', R)) - set(V))
assert not missing, missing
out = R.format(**V)
os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, 'w').write(out)
print(len(out.split('\n')), 'lines written to', DST)
