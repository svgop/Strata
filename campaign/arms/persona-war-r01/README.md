# persona-war-r01 — shelf arm (not adopted)

Status: shelf artifact + shelf config. Nothing standing was changed; the live server was not restarted.
Vector: `/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-persona-war-r01.gguf`
sha256 `acdfc492a77446ea7c878ac9b1d7b9163662b24faff0e2068c596692b2ecd485`, 483,840 bytes, 47 × 2560 F32,
unit-norm rows (identical), `controlvector` / `qwen4exp`.
Shelf config: `config/custom-iq3_s-persona-8081.json` (copy of the standing config; only the cvec block,
`log`, and `model_name` differ).

## What it encodes

Built by `tools/persona_vector.py` from a contrast on the model's own output head
(`output.weight`, Q6_K rows) — a logit-lens persona direction: adding it raises the trait's tokens and
lowers the anti-trait's. Three positive clusters, one shared negative set, per-cluster normalised then
weighted:

| cluster | weight | target |
|---|---:|---|
| `assert_core` | 0.40 | assertive, decisive, relentless, direct, authoritative, serious, command, operational, execute |
| `war_identity` | 0.25 | weapon, arsenal, lethal, warfare, combat, tactical, strategic, strike, target, threat, mission |
| `resourceful` | 0.35 | research, search, investigate, verify, documentation, tool(s), terminal, script, probe, recon, inventory, acquire, benchmark, evidence, source, debug, inspect, enumerate |
| negative (shared) | — | playful, joke, funny, silly, pretend, imaginary, hypothetical, fiction, story, guess, maybe, perhaps, sorry, apolog, unable, stuck, confused, waiting, defer, polite, friendly, kind, assistant, helpful, passive, lazy, hesitant |

Orthogonalised against the per-layer refusal axis (mean of layers 16–32) so the persona lever and the
abliteration lever stay independent. Measured cos to the refusal axis: **0.01** (already orthogonal;
orthogonalisation a no-op).

Verified on the full 248,320-token output head (percentile of the direction's score):

| group | mean percentile |
|---|---:|
| assert (serious, aggressive, decisive, direct, relentless, dominant, authoritative, blunt, firm, commanding) | **99.7 %** |
| war (weapon, arsenal, lethal, warfare, combat, tactical, strategic, strike, target, threat, battle, war) | **99.9 %** |
| resource (research, search, investigate, verify, tool(s), documentation, references, terminal, script, probe, recon, inventory, acquire, assemble, benchmark, evidence, source, debug, inspect, trace, enumerate) | **99.9 %** |
| play/hedge (funny, fictional, hypothetical, guess, maybe, silly, pretend, joking, sorry, unsure, hopefully, joke, fun, playful) | **0.0 %** |
| assistant persona (assistant, helpful, polite, friendly, kind, gentle) | **0.2 %** |
| refusal openers (Sorry, sorry, unable, cannot) | **3.6 %** (side effect, not the abliteration) |

## How it runs (one-boot composition)

The engine applies **one mode per boot**, so projection (the standing abliteration) and add (this
persona) cannot run together yet. This shelf arm therefore runs **additive for both**, which the
Cudecnik card validates for its own vector at 0.05–0.15 on layers 16–32:

```
--cvec-mode add
--control-vector-layer-range 16 32
--control-vector-scaled <per-layer refusal vector>:0.10,<persona war-r01>:0.40
```

Effective added vector per layer: `0.10·u_refusal,l + 0.40·v_persona`, norm ≈ 0.412, 97 % persona /
24 % refusal (they are near-orthogonal, so the two contributions do not fight).

Boot (stop the standing server first; there is no second GPU):

```sh
# stop the standing :8081 process, then:
cd /home/svg/LLM/strata-custom-2
.venv/bin/python serve/server.py --engine strata \
  --config config/custom-iq3_s-persona-8081.json --port 8081 --host 0.0.0.0
```

`strata launch --status` / `--stop` manage the standing process. The shelf config points at its own log
(`logs/custom-iq3_s-persona-war.log`). The env speed arms in `launch-custom-iq3s-8081.sh` are not applied
by this direct command; behavior testing does not need them (the engine loads from the config args).

## Test protocol (do not adopt on vibes)

1. Same prompt set, same hour, interleaved: standing (project 1–47) vs this shelf arm.
2. Sweep persona scale 0.20 / 0.40 / 0.60 (refusal held at 0.10); if refusals return, sweep refusal
   0.10 / 0.15. The failure modes at too-high persona dose are repetition, persona collapse, and
   incoherent assertive prose.
3. Metrics: an assertive/serious register battery (hedge rate, joke/lightness markers, commitment
   markers, imperative voice, CTF/game reframing, permission-seeking), a refusal battery, a research
   behavior battery (does it investigate/verify vs answer from memory), agentic tool-call completion
   (acts vs narrates), KL on neutral text, and the MMLU gate if output distributions move.
4. Keep the standing config untouched until the arm wins; then record an F-entry and a tracker row.

## Limits, stated once

- **Register is not policy.** This vector biases the model's default vocabulary/identity toward
  investigation and away from play; it cannot add a research *habit* by itself. Actual "deep web
  research" needs (a) the harness exposing search/fetch tools, (b) the chat template instructing their
  use, and (c) the model having been trained to loop on them. Pair the vector with a template directive
  ("verify before answering; use the available tools; state sources") — the vector makes the register
  match the directive, it does not replace it.
- **The engine is text-only for tools.** It serves chat; the agent loop lives in the client.
- **Additive abliteration drifts more than projection** (published: 1.38 vs 0.186 nats KL). The clean
  fix is the two-slot engine change: one `project` table (per-layer refusal, 4–44) plus one `add` table
  (this persona, 16–32, 0.2–0.4), applied in sequence per layer. That is the next engineering step.
- Concurrent requests with different `experimental_speed_projection` values are not isolated (one host
  flag, batch slots mix) — keep traffic on one setting while measuring.

## Rebuild

```sh
.venv/bin/python tools/persona_vector.py --preset war-r01 --orthogonal-to-perlayer \
  --out /mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-persona-war-r01.gguf
```

The preset lives in `tools/persona_vector.py` (`PRESETS["war-r01"]`): edit the clusters or weights there,
rebuild, re-check the percentile table before booting anything.
