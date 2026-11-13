# Frontier prediction-theory audit (2026-11-13) — branch prediction / hardware prefetch / online learning vs RouterLookahead

Batch 3 of the 18-agent frontier wave.

## Hypothesis CORRECTED
Our machinery is THREE predictor classes, none a Markov chain: (1) CAUSAL EXACT RECOMPUTE (dominant) — the
real next-layer gate x the real current input = Belady-oracle at depth 1-2 (RouterLookahead
expert_source.cpp:2961-2985; DS4 route_ahead on-device); error only via the 8-token subsample and depth>1
stale-input; (2) STATIC MARGINAL — the learned profile = TAGE's bimodal fallback table; (3) USEFULNESS
CREDIT — PREDICT_CREDIT/heat = SHiP counters (replacement state, not prediction). Warm 97-99% comes from
class 1's exactness, NOT learning. DS4 per-rank recall 97/91/81/70/57/41%.

## The A2 bound read correctly
The flat 4th-pick tail bounds ONLY the marginal class (~11-15% per pick). It does NOT bound
history-conditioned predictors — the ceiling is the conditional entropy H(E_t|E_{t-1..k}); branch
prediction's own history is the proof (flat marginals, 99%+ under TAGE). BUT the honest caveat: our exact
recompute already owns warm decode; TAGE-class machinery can only buy depth>2, gate-closed fallback,
prefill subsample error, and churn windows. MEASURE the conditional entropy on DS4_TRACE before building
anything (one scripting day vs weeks of porting; the trace format already dumps routing+guesses).

## Instrumentation gaps (log before building)
1. Conditional-entropy sweep k=1..8 per layer (the TAGE go/no-go). 2. Timeliness CDF issue->arrival->use
(only booleans exist: pf_late/ahead_cold_hit). 3. Belady offline bound on traces — split the 1-3.5% miss
pool into eviction-race vs mispredict vs beyond-depth; also split st_actual_ by cause. 4. Prefill
subsample agreement. 5. Pollution + prefetch:demand byte ratio. 6. pf_late confounds "arrived late" with
"never needed".

## Ranked findings
F1. Hedge/tournament over the predictor pool (regret sqrt(T ln N/2)) — per-(layer,depth) exponential
weights on {recompute, profile, future members}, fixed-share rho 1e-3 for churn; replaces hand constants
0.5^dj and io_minconf_; LOW port, inits to current behavior, protects the warm band through churn.
F2. Belady/Hawkeye offline bound FIRST + SHiP-formalized credit (set margin/bonus from MIN samples instead
of constants) — free; likely most of the residual is replacement, not prediction. DO BEFORE F3-F5.
F3. TAGE-class tagged tables over routing-signature history (k in 1..16, bimodal fallback = the profile) —
only if gap 1 shows signal; MED-HIGH port; never displaces exact recompute (backfill only).
F4. Markov/delta-correlation for the churn phase (Chen-Baer/Joseph-Grunwald/BOP; timeliness-first scoring
replacing fixed 0.5^dj).
F5. Perceptron on the stale-input feature for depth 2-4 (x_ is already the feature vector; targets the
97->41 per-rank decay).
F6. DIP/BIP insertion policy — prefetched lines insert cold (churn-safe aggression; the rank>0 credit
halving already encodes the intent).
F7. Sandbox prefetching (shadow verification before admission) — enables aggressive predictors while
holding "prefetch never hurts".
F8. Pythia-style reward as the principled PREDICT_CREDIT (per-candidate action scores under the byte
budget) — after gaps 2-3; HIGH port; reward-hacking risk (the bandwidth term is load-bearing: the lane is
DRAM-saturated, only fewer/better-placed bytes move it).

## Order of operations
F2 (free bound) -> gaps 1-2 -> F1 (small robustness) -> F6+F7 (safe aggression) -> F3/F4/F5 ONLY if the
entropy sweep shows history carries signal.

Sources: Seznec-Michaud TAGE; TAGE-SC-L; Yeh-Patt; Jimenez-Lin perceptron; MWU/Hedge (Freund-Schapire,
Mourtada-Seldin); Chen-Baer; Joseph-Grunwald; Michaud BOP; PPF; Sandbox HPCA'14; DIP Qureshi; SHiP;
Hawkeye/Belady-ML; Pythia MICRO'21; FDP HPCA'07; Pre-gated MoE; MoE-Infinity; LWN readahead; mm/readahead.c.
