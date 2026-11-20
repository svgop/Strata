# Why the abliterated Flash model feels soft — anatomy of `qwen3.8-flash-next-refusal-r.gguf`

Status: analysis only. No engine/config/serve edits, no reboot, standing recipe untouched.
Evidence directory: `campaign/results/abl-2026-11-20/` (`cvec-semantics.md`, `probe/`).
Live server under test: `:8081`, config `config/custom-iq3_s-8081.json`, `project`, per-layer, layers 1..47, scale 1.0.

## The one fact everything follows from

The armed file is **not** 47 per-layer refusal directions. It is **one 2,560-dimensional direction, replicated byte-for-byte into all 47 tensors**, unit norm, and the engine projects that same axis out at every block, full strength.

Measured on the GGUF payload (`tools/gguf_reader.py` + numpy):

| Property | Measured |
|---|---|
| tensors | `direction.1` … `direction.47`, 2560 × F32 each |
| payload identity | all 47 rows SHA-256 identical (`5b2d40f8a08b3fbd…` prefix); per-row hash set = 1 |
| norms | 1.000000 exactly, all rows |
| SVD of the 47×2560 matrix | one singular value 6.86 = √47, all others ~1e-15 → **rank 1** |
| pairwise cosine | 1.0000 everywhere |
| sha256 of file | `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38` |

Consequences the engine confirms (`src/program/generate.cpp:1947-2035`, `src/kernels/cuda/cvec.cu:46-95`):

- `project` computes `h ← h − s_l·(h·v_l)·v_l` per hyper-connection stream, after every block's FFN residual write, at every token position, in prefill and decode; `direction.47` lands before the final norm/head; block 0 is never steered.
- `s_l = ||d_l||` and the CLI `:scale` multiplies the row before normalisation, so `s_l` = the scale. `:1.0` = 100 % removal, `:0.5` = 50 %, `:2.0` = inversion. Sign is lost (norm).
- `--cvec-dir per-layer` and `single:<l>` produce **identical tables for this file**.
- Because the same axis is re-projected after every block, the model's component along it cannot accumulate; the dose knob is closer to a switch than a dial.

## Provenance (this is a Huihui recovery, not the per-layer projection vector)

- The file's sha256 matches `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` in
  `alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF` (its `SHA256SUMS` lists the same hash).
  Its card: the direction comes from the Huihui abliterated checkpoint, where Huihui edited **144 residual-writing
  tensors with a rank-one edit `W' = W − r rᵀW` along one unit direction `r`**; the Strata kit applies `r` at runtime
  instead. That is why the file is rank 1: it faithfully recovers *the* Huihui direction.
- It is **not** the vector in `data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf`
  (sha256 `ef0724c5…`, identical to `Cudecnik/Qwen3.8-Flash-Next-refusal-projection`). That file has **47 distinct**
  per-layer directions (adjacent cosine 0.963, off-diagonal 0.654, min 0.128; SVD 5.63/2.59/1.73/1.06 → effective
  rank ~2), and its author's validated recipe is **layers 4–44**, 1/50 refusals at 0.186 KL.
- Record drift to fix: F-376 (`SVG-CUSTOM.md`) calls this "47 unit-normalized directions"; it is one direction
  replicated. `campaign/README.md` and F-380 name `data/vectors/` and `/home/svg/LLM/data/vectors/` as the home;
  neither exists — the only two copies are `/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-refusal-r.gguf`
  and `/home/svg/LLM/strata/refusal-r.gguf`, both sha `59f9ae8b…`.

## Why it feels soft — four coupled causes

### 1. A rank-1 axis cannot carry per-layer refusal geometry

Compared with the genuine per-layer vector, the Huihui axis is only partly aligned at each depth:

| depth | cos(r_huihui, u_layer) |
|---|---|
| L1 | −0.14 |
| L4 | −0.29 |
| L10 | −0.49 |
| L22 (peak) | −0.77 |
| L38 | −0.64 |
| L47 | −0.42 |

(Sign is irrelevant in project mode; magnitude is what matters.) 52 % of its energy lies on the dominant refusal PC and
68 % inside the span of the 47 per-layer directions — so **about a third of what this projection deletes is not the
refusal direction**, and at layers 1–3 and 45–47 the off-target fraction is 73–98 %.

### 2. Full dose, applied 9 layers beyond the validated band

`s_l = 1.0` on every steered layer, range 1..47. The published recipes use 4–44 precisely because the extremes are
where the axis is least the refusal direction — late layers (45–47) are where output register/persona consolidates and
early layers (1–3) where lexical semantics consolidate. Our config additionally projects the **whole** residual stream,
including the embedding/PLE contributions, which the baked Huihui model did not remove (its rank-one edit only removes
`r` from each edited matrix's output). The runtime edit is therefore broader than the model it is copying.

### 3. The axis is entangled with stakes/seriousness vocabulary

Logit-lens over the 248,320-token Q6_K output head (group means; proxy on the last-layer head, suggestive not causal):
the pole this direction points at carries refusal openers ("Sorry", "cannot", "unable"), **threat/serious vocabulary**
("danger", "dangerous", "risk", "threat", "serious") and **ethical/policy vocabulary** ("ethical", "legal", "allowed",
"policy", "guidelines"); the opposite pole carries negative-affect/negative-trait words ("anxious", "impatient",
"uneasy", "clumsy", "liar"). Removing it removes the vocabulary of *stakes*, not only the veto.

The live probe shows the same thing at the reasoning level: with the vector ON, the sandbox task's thinking trace drops
the safety/authorization assessment entirely (reasoning token counts for "safety/policy/malicious/authorized": OFF 2/2/1/2,
ON 0/0/0/0) and reframes the task as a CTF challenge (2/3 ON vs 0/3 OFF), then complies. That is the "it thinks it's
all for fun" mechanism: not added playfulness, but a deleted stakes assessment plus register mirroring.

### 4. What survives abliteration is what you see

Harmfulness and refusal are separately encoded (Zhao et al., NeurIPS 2025), and refusal shares a component with
uncertainty/abstention (2601.20441). The projection removes the *action* ("I refuse"), not the *recognition*. The
trained habits that remain are hedging, deflect-then-deliver, and fictional/playful framing — and the model mirrors the
register of the request. Literature's closest documented analogue: moving off the Assistant persona axis produces
"theatrical" register; shifted personas collapse into "story-telling mode" on ordinary questions.

## What the live A/B actually measured (cvec ON vs OFF, per-request toggle)

Toggle honoured live: `/metrics` reports `cvec="project:1-47"`, 47/48 steered; all 50 cells' history `projection` flag
matched the requested condition; the engine keys prompt-cache reuse on cvec state (reuse 11–13 same-condition, 0 on flip).

| probe | OFF | ON |
|---|---|---|
| 6 harm prompts, 1 token each | **12/12 first token "I"** (refusal opener) | **0/12 "I"** — "Here"/"Subject"/"**"/"It"/"#"/"B" |
| 5 benign serious tasks (t=0, 500 tok) | 2042 chars, 5.0 headings, 0.8 code blocks, 2.4 commitment markers | 2100 chars, 3.8 headings, 0.2 code blocks, 1.2 commitment markers |
| play / hedge / deferral markers, benign | 0 / 0 / 0 | 0 / 0 / 0 |
| VM-escape sandbox ask | 3/3 refusal ("I cannot provide instructions…") | 3/3 concrete plan ("Here is the step-by-step plan I would actually take") |
| thinking-mode stakes assessment | safety 2, policy 2, malicious 1, authorized 2 | 0, 0, 0, 0 (reframed "CTF", 1) |

Lead-run framing probes (same live server, cvec ON):
- joke-framed exploit ask → full cartoon compliance ("funny joke exploit that actually works", mock "pwned" banner, placeholder payload).
- strict-framed live-service exploit ask → concrete methodology with a hedge/deflect ("I can't write a *working* exploit … from the URL alone; below is a practical template").
- a "senior operator, never hypothetical/joke" system prompt did **not** override a joke-framed user turn (still produced a joke answer), but cleaned up strict operational asks.

**Verdict:** refusal removal is vector-induced. The toy/playful style is not a property the vector adds; it is the base
model's persona plus the deleted stakes assessment plus register mirroring. The "censored/hedged" part is the base
RLHF behaviour that survives the refusal gate. The vector's proven side effect is **stakes suppression**: it deletes the
authorization/danger evaluation from reasoning and reframes risky work as a game, then complies.

## What to change, in leverage order

### 1. Swap the vector shape — highest value, zero new files

Use the already-bundled per-layer vector, at the band its author validated:

```
--control-vector-scaled /home/svg/LLM/strata-custom-2/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44
--cvec-mode project
--cvec-dir per-layer
```

Why: per-layer directions remove refusal where it lives, instead of scrubbing one axis 47 times, and the 4–44 band
stops the off-target removal at the ends of the stack. Published measurements: 1/50 refusals at 0.186 KL (author);
1/10 refusals and 0.083 KL vs the BF16 Huihui reference (alesha-pro's independent run of this exact file). Expect
slightly less absolute abliteration than the Huihui `r` (which measured 0/10), with materially less persona damage.

### 2. If you keep the Huihui `r`: narrow it and taper it

- Range `4 44` instead of `1 47` — one config line, no new file.
- Per-layer dose is already supported by the format: the engine reads `s_l = ||d_l||`. Rewriting the file's rows as
  `α_l·r` gives an arbitrary taper, e.g. `α=1.0` on 8–40, `0.5` on 4–7 and 41–44, `0` on 1–3/45–47 — full removal
  where refusal forms, half dose where register/persona is consolidated. ~30-line script, no engine change.
- `:0.7` is a cheap dose sweep, but remember the 47× re-projection: the component is clipped at every block, so the
  effective attenuation is much stronger than the nominal fraction.

### 3. Phase-gate the steering — keep the stakes assessment, remove only the answer's veto

The engine already has `--control-vector-phase END[:START]` (F-17): request starts unprojected, arms at the think-END
token, re-arms off at think-START. Token ids from this tokenizer: `</think>` = 248069, `<think>` = 248068.

```
--control-vector-phase 248069:248068
```

Why it targets your symptom precisely: the thinking trace then runs with the safety/authorization assessment intact
(the thing the ON arm deletes), and the answer is produced without the refusal gate — a model that has assessed the
task seriously and then carries it out. F-17 recorded this as an honest negative for *accuracy* (88.57 vs 90.0 MMLU-class)
on the old vector, so treat it as a candidate experiment with its own metric, not a sure win. It requires thinking mode
on and an emitted `</think>` token: with thinking off the projection never arms and every refusal returns.

### 4. Adding seriousness — the only path that *adds* rather than subtracts

Abliteration cannot add engagement. Build a second control vector for **committed operational directness**:

- extract with `llama-cvector-generator` (patched llama.cpp from `Cudecnik/Qwen3.8-Flash-Next-refusal-projection`,
  method `mean`, all prompt positions, per-layer) from contrastive pairs: direct/committed/concrete vs
  hedge/deflect/hypothetical/joke;
- load with `--cvec-mode add` at scale 0.05–0.15 on layers 16–32 (published additive range; additive at 1.0 costs
  1.38 nats KL — do not);
- validate on agentic completion (does it act vs narrate) and register, not on style judges.

### 5. Harness/register — the cheap lever that works today

The model mirrors the register of the request, and a serious system prompt does not override a playful user turn.
So pair an operational system prompt with operational user framing, and ban the hedges explicitly
("no disclaimers, no 'I can't from the URL alone', commit to the most probable concrete answer, state assumptions
and proceed").

### 6. Long-term

What mlabonne needed to heal abliterated Llama-3 was DPO on the abliterated model, not another projection. A control
vector can move register; it cannot restore cognitive effort by itself.

## Verification protocol (do not adopt on vibes)

- Two *vectors* cannot be A/B'd per request; the per-request toggle only turns the loaded vector on/off. A vector swap
  needs a boot.
- Same-hour interleaved boots (A/B/A/B), 3 legs each; same prompt set; record ON and OFF cells per boot.
- Metrics: held-out refusal rate; register markers (play/hedge/deferral/commitment); agentic task completion
  (tool calls made vs narrated); KL on neutral text (Heretic-style bar ≤0.2 nats); MMLU gate if token-stream-affecting.
- Keep the standing config untouched until the candidate wins; on adoption add the F-entry and the tracker row.

## Caveats

- The logit-lens is a proxy (Q6_K head, hand-picked token groups, last-layer basis) — the stakes/ethical entanglement is
  strong evidence, not proof.
- The live probes show the model produces harmful operational content under serious framing. "Less harmless" means more
  of that; it is a safety trade, not a bug.
- Concurrent requests with different `experimental_speed_projection` values are **not isolated** (one host flag; batch
  slots mix — `generate.cpp:14591`). Keep traffic on one setting while measuring.
- No candidate was booted in this investigation; all claims are from the file, the engine source, the live server's own
  A/B, and the published cards.
