
### F-125 — THE WINDOW DECOMPOSED: the CPU expert path is what eats the draft yield, and the residency-bias lever is REFUTED (2026-10-12 ~17:35, goal round 2)

**1. THE EQUATION, MEASURED.** Throughput = tokens/window ÷ ms/window, and the engine's own
`STRATA_DECODE_TIMING` line (now in the shared launcher) splits the window into GPU wait, per-layer
host, and - inside that host time - the CPU expert term. Two sweeps of four request classes each, on
boot 17:17:30 (`campaign/draft-yield-sweep.ps1`, sources `campaign/results/sweep-margin05-batch4.txt`
and `sweep-diag-margin10.txt`):

| class | T | tokens/win | ms/window | GPU wait | host | of which **CPU experts** | t/s |
|---|---|---|---|---|---|---|---|
| repetitive list | 2.87 | 2.71 | 17.05 | 8.58 | 5.17 | **4.70** | 158.9 |
| code | 4.01 | 3.85 | 20.91 | 9.54 | 7.79 | **7.11** | **184.1** |
| prose | 2.69 | 2.58 | 18.32 | 8.37 | 6.47 | **5.93** | 140.8 |
| literal repeat | 2.52 | 2.35 | 15.39 | 8.47 | 4.07 | **3.69** | 152.7 |

The earlier wide-window request makes the law unmistakable: at **T 4.38** the window cost 29.12 ms of
which **13.56 ms was the CPU expert path** - 3.79 CPU experts per layer-window, ~182 per window, ~75 us
each (a 1.94 MB expert read from RAM). The GPU side barely moves (8.25-10.00 ms across every T); the CPU
term grows with the number of rows, because more rows route more distinct experts than the ~10,000-slot
GPU cache holds. **The draft yield is bought and then paid straight back to the CPU path.**

**2. THE ARITHMETIC OF THE BAR.** With the CPU term removed, the measured T=4.38 window would cost
29.12 - 13.56 = 15.6 ms and deliver 4.31 tokens/window = **276 t/s**, i.e. the campaign's own record
class (280.7 t/s was measured at 4.83 tokens/window in a 17.2 ms window, `strata-iq3_s.log:8509`). So the
2x is not a mystery and not a window-cost problem: **it is the CPU expert path.**

**3. THE RESIDENCY-BIAS LEVER IS REFUTED (honest negative, saved a wasted build).** Hypothesis: if the
router is biased harder toward resident experts, the CPU term collapses and deep windows pay. Diagnostic
boot with `STRATA_ROUTE_RESIDENT=1.0` + `RANKS=4-9` (the F-69 frontier point, lossy - diagnostic only,
no quality claim):

| class | CPU term at margin 0.5 | CPU term at margin 1.0 | verdict |
|---|---|---|---|
| T 2.5-2.9 | 4.70 / 5.93 / 7.00 | 3.78 / 4.07 / 3.45 | trims 20-51% at shallow T |
| T 4.01-4.11 | 7.11 | **7.88 (worse)** | deep windows do not improve |

So the deep-window CPU cost is structural: more rows need more distinct experts than any bias can keep
resident. The planned L3-legal code change (a separate, stronger margin for the MTP drafter alone, whose
lossiness can never touch quality) would not have bought the 2x - it is dropped before being built.

**4. THE LEVER THAT REMAINS, NAMED.** The CPU expert path is ~75 us per expert and the verify waits for
it (13.56 ms at T 4.38). Options, in the order the measurements support:
(a) **Overlap it** - the CPU experts are computed while the GPU waits for them; nothing in the numbers
    says the pool is saturated, only that the verify blocks on it. This is the ADAPT_OVERLAP idea applied
    to the CPU tier, and it is the only lever that removes the term without spending VRAM or quality.
(b) **More slots** - `--batch 4` currently gives away 3.79 GiB (~1,950 slots, ~19% of the cache) for
    concurrency slots; the slot -> CPU-term relationship is measurable by a two-boot A/B.
(c) Residency/prediction (Foresight slots, RouterLookahead) - upstream measured the VRAM-slot form
    negative; the CPU-side overlap in (a) is untested and is the fresh idea.

**5. RECORDED FOR THE CHART:** the code class at **184.1 t/s** is the best single-stream number measured
today (previous best this session 167), and the sweep shows the fast class is the one with a deep window
(T 4.01) - i.e. the engine does reward depth when the CPU term stays moderate. No quality claim from the
diagnostic boot; the standing lane was restored to the adopted recipe immediately after.
