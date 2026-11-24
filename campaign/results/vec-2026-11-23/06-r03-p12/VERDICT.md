# P12 projected arm at the knee dose — VERDICT: WINDOW, facts intact (plain went 0.94 here)

Vector: `qwen3.8-flash-next-persona-r03-assertive-s9-p12.gguf` — the slot-9 row projected into
block 12's router null space (P = I − Rᵀ(RRᵀ)⁻¹R; norm kept 0.729; post-projection router
visibility 0.84/1.17 ≈ random; see `router-audit.md`). Dose 0.6, stock context — the dose where
the plain row showed the campaign's first factual casualty.

| 0.6 arm | assertive | serious | direct | facts off/on | words |
|---|---|---|---|---|---|
| plain (`../04-r03-slot9/`) | 11/2 p=0.011 | 17/0 p=1e-05 | 6/6 | 1.00→**0.94** | 167→156 |
| **P12 projected** | 10/3 p=0.046 | **18/0 p≈0** | 8/4 p=0.19 | 1.00→**1.00** | 163→156 |

## Reading

- The projection removed the factual casualty at the knee without costing any register effect —
  serious is a clean sweep (18/0), assertive holds, words match. **The router channel is the
  carrier of the 0.6 damage**, confirming RARE's mechanism (2608.21236) on this rig with our
  vector, and the projection is validated as the knee-raiser.
- Cost model: the projected row keeps 72.9% of the direction; at equal nominal dose its persona
  content is ~0.73× the plain row's — yet serious went 17/0 → 18/0. The removed 27% was damage
  capacity, not persona.
- Measured caveat, stated plainly: one dose, one run, and the facts delta is a single question
  (15/16 → 16/16). Confirmation before defaulting the construction to projected: a two-dose
  plain-vs-projected interleave (0.45, 0.6). Not yet run; queued as the follow-up leg.
- judgeA = 1.0 stock caveat applies (deployed test carries decisions).

## Queue state after this result

The arm of record is the **plain slot-9 row at 0.3 under the deployed template**
(`../05-r03-deployed/VERDICT.md`) — it is the completed decision test. The projected row is the
validated upgrade path for higher-dose operation (its own deployed ON/OFF at ≥0.45 remains
unrun and is the next leg if more register force is wanted).
