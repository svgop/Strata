# campaign/research INDEX

| date | file | topic | conclusion | status |
|---|---|---|---|---|
| 2026-10-16 | T15B-262k-switch-plan.md | --max-context 131072→262144 switch: trace, diff, test plan, risks | one-line json diff (custom-iq3s-262k.json already matches); c7544ef6 port NOT needed (our kernel has capacity-sized dynamic smem + fallback; 262k AND 1M boots ran this kernel Oct 14); costs: +224 MiB/session → ~−700 expert slots, pinned KV 1.55→3.09 GiB, kvg_ensure stays OFF, conv cache unchanged; F-182 A/B = noise at short usage, F-181 = −11% decode at 244k depth; GO conditional on 3v3 conc + MMLU floor; trap: stale 262k launcher env | current |
| 2026-10-17 | T15C-overlooked-wins.md | never-swept env inventory (417 knobs), overlooked wins, stale defaults | S23/S26 decode-kernel package (Q6_PACKED/TSUM/HC_Q8) never armed here; LANECELL×DAILY×COHORTS never separated on conc; L2-72 + DT peak + pcie-frac 0.35 untested on true base; 10 ranked candidates, 6 env-only boots; sampler double-sync unexecuted; GDN step ILP rewrite open | current |
| 2026-10-17 | T14C-fastmtp-acceptance-math-gates.md | FastMTP drafter retrain: acceptance→t/s model, integration gates, literature | c_row≈3.56ms/row (M1)/4.62 (conc); retrain alone = +5.7% M1/+2.8% pair/+2.7% conc; depth pays only >0.72 tokens/extra-row; 250 needs tranche AND retrain; gate ladder G0-G5 with cheap falsifiers | current |
| 2026-10-16 | T13A-decode-critical-path.md | serial decode window anatomy and lever ranking | window 11.4-12.0ms @ 2.47 tok; micro-tranche −0.9..−1.7ms; depth lever premise later re-priced by F-228/T14C (marginal row ≠ 0.55ms) | current (T14C amends its depth-EV arithmetic) |
| 2026-10-16 | T13B-concurrency-scheduling.md | batch scheduling lanes | — | current |
| 2026-10-16 | T13C-memory-prefill-frontier.md | memory/prefill frontier | — | current |
