# T14B — HARVEST EXECUTION spec: hidden-state dump + Draft-OPD replay corpus for the FastMTP retrain (r18)

Date: 2026-10-18. Owner lane: T16 W2 / T17 #3 "RETRAIN v2". This is the execution spec for the HARVEST —
it produces the trainer's input data (`r18-train/records/` + `r18-train/opd/`); the trainer consumes it per
`campaign/T14B-specs/trainer-spec.md` §3/§8, the gates are T14C's ladder (gates-spec.md). No new research
performed; every external claim cites a banked dossier or arXiv ID.

Banked inputs (read this session):
- `SVG-CUSTOM-TODO.md` T11.R3 (:793-806), T17 QUEUE (:1096-1236), T17 CONSOLIDATED VERDICT (:1215-1236),
  STEP 0 RESULTS (:1238-1269).
- `SVG-CUSTOM.md` F-235 (:8074-8102), F-236 (:8104-8137) — the depth-split finding (data-spec input).
- `campaign/research/T14A-fastmtp-method-recipe.md` (arms, risks, F1/F2 data laws).
- `campaign/research/T14B-fastmtp-training-infra.md` (P1 harvest plan, dump inventory, F-77..F-118 laws).
- `campaign/research/T17C-spec-algo-leaps.md` (§2 contract map, §5.1 warm-start/data-scale).
- `campaign/research/T17F-training-quant-context.md` (§3.1 Draft-OPD replay, §3.4 LongSpec aug, §6 sequencing law).
- `campaign/T14B-specs/trainer-spec.md` (the downstream consumer; its §3 contract is honored below).

Hard constraints honored: read-only session except this file; no builds, no engine boots, no server calls,
no git operations. All code claims below were verified by reading the current tree (`engine` = this repo);
all rig facts re-verified 2026-10-18 (df, ls, mounts).

---

## 1. Summary

The harvest = TWO passes over ONE assembled corpus, plus a ~half-day of env-gated host-side instrumentation:

- **PASS A — self-distill + Draft-OPD decode pass** (~2-2.5 h): drive seed documents through the standing
  recipe as chat conversations, generate the target's own replies (FastMTP sampler: temp 0.6, top-p 0.95),
  with the coupled drafter live, logging every (position, draft, accepted/rejected, target pick) tuple.
  This is the Draft-OPD rejection-replay dataset (arXiv 2605.29343; T17F #2) and the self-distilled text
  the corpus is made of (T14A F1/F2: "MTP requires training data that the target model itself generates").
- **PASS B — prefill-only hidden-state harvest** (~45-60 min): replay each finished conversation as a fresh
  request with `max_tokens=1` (the campaign's measured prefill-only procedure, campaign/prefill-probe.sh) and
  harvest EVERY prompt position's final residual R — the drafter's exact serving input — with the in-tree
  env hook `STRATA_PREFILL_DUMP_R_ALL` (src/prefill/prefill.cpp:4193-4214), bf16, batched-path, full prefill
  rate (~6.3-7.2k tok/s measured).

**Headline code facts this spec is built on (all read in-tree this session):**
1. **The batched hidden-state dump hook ALREADY EXISTS.** `STRATA_PREFILL_DUMP_R_ALL=<path>` writes every
   position's final multi-stream residual as bf16 (round-to-nearest-even), rows in position order, appended
   across chunks and requests, shape [n][hc*n_embd] = [n][10240] — commented in-tree as "S25 (draft-layer
   distillation data, opt-in)" (src/prefill/prefill.cpp:4193-4214). The T14B dossier priced this as a ~50-line
   extension to build (T14B §3 P1, §6 risk 3); it is shipped. ZERO engine code is needed for the R dump.
2. **`--dump-final-r` alone cannot harvest a corpus on our pack.** It writes only in the per-token drive loop
   (generate.cpp:13969-13974); on our native (IQ) pack the whole prompt is FORCED through the batched path
   (`--prefill CHUNK` is required for native packs, generate.cpp:3117-3123), so per request it yields only the
   last prompt position + generated tokens. It stays in the spec as the numerics cross-check probe (§5.5) and
   as the trainer-record FORMAT donor — not as the harvest mechanism.
3. **One pass cannot produce both artifacts.** Rejection tuples exist only in decode (spec) windows; a
   prefill-only pass (`max_tokens=1`) never rejects. The task's "the harvest pass should ALSO log rejection/
   pick pairs" is resolved as two passes over one corpus (§4, §5): Pass A decodes (and therefore drafts, and
   therefore rejects), Pass B prefills. Each pass is cheap; together they fit one overnight slot.

---

## 2. Data contract (what the harvest emits, and who reads it)

Artifacts land under `/home/svg/LLM/Custom-data/mtp-iq3s/r18-train/` (the trainer-spec §8 layout):

```
r18-train/
  records/
    raw-r.bf16        PASS B raw S25 dump: headerless rows [n][10240] bf16-RNE, position-ordered,
                      appended across chunks and requests (prefill.cpp:4193-4214)
    ledger.jsonl      one line per request: {doc_id, band, byte_start, byte_end, rows, prompt_tokens}
                      — THE alignment artifact (the dump carries no per-row keys)
    records.f32       packed trainer input in trainer-spec §3.1 format: per position,
                      int64 pos (doc-local), int64 tok, then 10240 x f32 R  (40,976 B/record),
                      produced by the packer from raw-r.bf16 + ledger + corpus token ids
    PROBE.md          probe results: dump rate, row-count contract check, finiteness, numerics cross-check
  opd/
    replay.jsonl      PASS A tuples (format §4.3), one line per verify window
    opd-rows.bf16     optional: the R row at each rejection position (20,480 B each), offset-referenced
                      from replay.jsonl (bounded: ~150k rows ~ 3.1 GB)
    acceptance-by-band.txt  per-request "drafts accepted X of Y" + avg-T lines parsed from the engine log
                      (the pre-retrain acceptance-vs-depth baseline the trainer's G0.2 read checks against)
  corpus/
    corpus.jsonl      the assembled conversations: doc_id, band, messages[] (exact sent JSON), rendered
                      prompt text, token ids (int32 array), seed provenance
    manifest.json     pack-stamp (pack file sha256s), config hash, launcher env hash, ladder histogram,
                      dates, engine boot command
```

Pair law (unchanged from T14B P1 / trainer-spec §3.1): record i holds the state after consuming token i;
the drafter's serving input at i is `(R_i, emb(tok_i))` and its label is `tok_{i+1}`; conversations stay
contiguous, cross-seam pairs are dropped (F-77 law). Rows per request = `prompt_tokens - 1` (asserted per
request from the server's `usage.prompt_tokens`; the probe confirms the constant, §5.4).

Width law: `R = hc x n_embd = 4 x 2560 = 10,240` values — the final multi-stream residual the drafter binds
to (`mtp.bind(wt, &native_head, ver.final_R_all(), ...)` generate.cpp:14157; the same object the on_chunk
callback receives as `R_rows` and feeds `mtp.prefill`/`draft_kv`, generate.cpp:8653-8677, prefill.hpp:49-56
and :141).

---

## 3. Corpus plan

### 3.1 Source survey (re-verified 2026-10-18)

| Source | Status | Role |
|---|---|---|
| MIMIC corpus (`~/LLM/Custom-data`) | **ABSENT — CONFIRMED AGAIN.** Custom-data holds only `gguf_fingerprints.json`, `models/` 78 GB, `mtp/` 6.5 GB, `mtp-iq3s/` 6.5 GB, `packs/` 1.5 GB (ls+du this session). Note: campaign lore's "mimic" = the mimicbench 64-prompt adherence benchmark (`campaign/mimicbench2.py`), not a corpus | none |
| `/mnt/DATA` (Windows frozen copies) | **NOT MOUNTED** — `/mnt/DATA` is an empty mountpoint; mounted NTFS volumes are `/mnt/HQ` (3.7 T, 2.4 T used), `/mnt/STORE`, `/mnt/SCRATCH` (mount/df this session). Read-allowed IF the operator mounts it; do not schedule against it | conditional only |
| Repo technical prose (the F-236 probe's proven corpus) | PRIMARY SEED. AGENTS.md + README.md + SVG-CUSTOM.md (8,136 ln) + SVG-CUSTOM-TODO.md + `docs/*.md` (40 files, 742 KB) + `campaign/research/*.md` (43 files, 758 KB) + `campaign/*.md` ~= 2.21 MB ~= **675k tokens** at the calibrated 3.27 chars/token (depth-lane-real.py:36). Real, diverse, human/agent-written; content-matched to this campaign's actual serving mix (F-236 promoted content-matched data) | seed text |
| `/mnt/HQ` (read-allowed, frozen) | SECONDARY SEED, <= 25% of corpus text. `store/KnowledgeBase/DATASET/` 4.5 GB (1,071 .md of 357,105 files — take md/txt only, 8-256 KB each, >= 90% printable UTF-8), `store/Repo Collection/` (repo READMEs/docs = code-adjacent prose). Diversifies domain mix (code/security/docs); gated by the one-doc A/B in §12 | optional diversifier |
| Self-distillation (Pass A output) | PRIMARY VOLUME. The target's own generations are the majority of corpus LABELS (FastMTP law, T14A F1: self-distilled beats fixed pairs 1.81x vs 1.67x; RH: "data that the target model itself generates") | volume + label distribution |
| `campaign/results/*.txt` (600 files, 4.8 MB) | EXCLUDED — measurement tables, not language; drafting-log text is off-distribution | none |
| r17-era `pairs-r17-*.bin` | NOT PRESENT (T14B §12 searched to depth 5; re-confirmed absent). Also excluded on principle: old-corpus conditioning is what F-104 convicted | none |

### 3.2 The depth ladder (position-oversampling scheme — the F-236 law)

F-236 measured, at ~121k real depth: deep-position acceptance 53.1% (635/1195) vs shallow 93.6% = **-40.5pt
POSITIONAL** (primary), and filler 68.0% vs real 53.1% at matched depth = **-14.9pt CONTENT**. The drafter
must therefore SEE deep positions, on real content, at far above natural traffic share. Standing serve
context = 131,072 (custom-iq3_s-8081.json `--max-context`), so the ladder tops at 126k (hard cap: doc total
= seed + instruction + reply + template <= 126k < 131,072 with generation headroom). The 262k ladder arm is
deferred to the W3 switch (T16 #6; deep-position training there rides LongSpec offset aug trainer-side,
T17F #3 — zero harvest cost).

v1 target: **8.0M rows** (~164 GB bf16, §6). Coverage is harvested ~1.3x OVER each band's share; the ladder
mix is ENFORCED AT PACK TIME by selecting doc spans into `records.f32` (rebalancing is then free; a harvest
re-run is not needed to re-weight bands).

| Band | Doc depth (rendered prompt) | Docs | Rows target (packed) | Harvest over-cover | Share |
|---|---|---|---|---|---|
| L0 shallow | 3-6k | ~500 | 2.4M | x1.3 | 30% |
| L1 | 8-32k (center ~20k) | ~80 | 1.6M | x1.3 | 20% |
| L2 | 32-64k (center ~43k) | ~28 | 1.2M | x1.3 | 15% |
| L3 | 64-96k (center ~80k) | ~15 | 1.2M | x1.3 | 15% |
| L4 deficit zone | 96-126k (center ~114k) | ~14 | 1.6M | x1.3 | 20% |
| **Total** | | **~640** | **~8.0M** | | 100% |

Rationale: L4 20% + L3 15% = 35% of rows at 64k+ positions ~= 7-30x natural traffic share (trainer-spec §3.1
asks ">= 2x"; the -40.5pt deficit and the parked 262k switch justify the excess; the pos-0 <= 1pp gate is
protected by L0's 30%). If G0.2 ever trips pos-0, the packer re-mixes toward L0 without re-harvesting.

Doc construction rules:
- Each doc = ONE conversation: `[user: <seed text> + <instruction>, assistant: <Pass A reply>]`; deep bands
  may add 1-2 follow-up turns (also Pass A products) to place more deep decode positions.
- Seed text per doc = a doc-specific ROTATED concatenation of pool slices (the depth-lane-real.py pattern:
  `===== FILE: <name> =====` markers, depth-lane-real.py:30-47). The pool (675k tokens) is smaller than total
  seed draw (~6.3M tokens across docs), so slices repeat ACROSS docs — but at DIFFERENT positions (rotation).
  This is deliberate: the F-105 position-locality law makes same-content-at-different-depths a feature
  (position invariance), and it is the data-layout analog of LongSpec's anchor-offset idea. It does mean
  document-level dedup (FastMTP's MinHash law) must operate on (content, position-range) pairs, not content
  alone; the packer records the rotation index per doc.
- Instructions are drawn from the campaign's actual request classes (technical continuation, summarize,
  explain, code-comment) — the serving distribution the drafter will see.
- Chat formatting: every doc goes through the SAME chat template the server applies
  (chat_template.jinja in the pack tokenizer dir, rendered by serve/server.py:3786-3798). The driver renders
  locally with jinja2 + the same template file and tokenizes with the SAME tokenizer module the server loads
  (`strata_tokenizer.ST.Tokenizer` over vocab.json/merges.txt/token_type.json, server.py:6703-6717) — driver
  token ids == engine token ids by construction, and the per-request length assertion (§5.4) proves it every
  request.

### 3.3 Corpus acceptance criteria (GO gate before Pass B)

1. Row total >= 7.0M harvested (>= 87% of plan); band histograms within +/-20% of ladder shares after pack.
2. Length-assertion pass rate >= 99% of requests (`driver_len == usage.prompt_tokens`); failures quarantined.
3. 100% of PACKED records finite (bf16->f32 scan); byte-ledger totals == raw-r.bf16 size minus quarantined spans.
4. Self-distilled share: >= 50% of packed rows' labels are target-generated tokens (Pass A replies).
5. L4 >= 1.2M packed rows at >= 96k positions; pos histogram recorded in manifest.json (trainer-spec §12
   asks for the histogram BEFORE training, not after).
6. Pack-stamp present: sha256 of both pack shards + config hash + launcher env hash (the harvest is
   distribution-coupled to the pack, §8).

---

## 4. PASS A — self-distill + Draft-OPD decode pass

### 4.1 Mechanics

- Vehicle: the standing launcher/config pattern (Launch Law), on a COPY: `custom-iq3s-harvest-8082.json`
  (identical args EXCEPT port 8082 and log path) + `launch-custom-iq3s-harvest.sh` = copy of
  launch-custom-iq3s-8081.sh:8-38 (all standing env arms verbatim — ATTN_DAILY, GDN_CHUNKED, FUSE_HITS,
  POOL_DIET, COHORTS=2, BATCH_MTP, SPEC_COUPLED, DRAFT_TEMP 0.85, ROUTE_RESIDENT...) with the OPD env added.
  Distribution parity is the point: the rejections must be the SERVED drafter's rejections.
- Requests: `/v1/chat/completions`, one conversation per doc-seed, one in-flight request at a time
  (the M1 lane; serial box law). Replies sampled at the FastMTP recipe settings: temp 0.6, top-p 0.95
  (T14A Arm 0 item 2; FastMTP F1 used temp 0.6/top-k 20/top-p 0.95).
- Deep bands: the first request of a deep doc prefills the full seed (the only full-price prefill); follow-up
  turns ride the parked conversation prefix (the conversation cache, standing `--prompt-cache` default 6) so
  follow-up decode happens AT depth without re-prefill — the F-236 probe's KV-reuse-continuation mechanics,
  now used to place deep OPD tuples cheaply.
- Per-request acceptance telemetry is free: parse the engine log's `DONE ... drafts accepted X of Y` line
  (generate.cpp:13518/:13599) and the `avg T` window line (:13364) per request into acceptance-by-band.txt —
  the same parse depth-lane-real.py:76-84 performs. This is the PRE-retrain acceptance-vs-depth baseline.

### 4.2 The Draft-OPD instrumentation (the ~half-day task; the only new code in the harvest)

Env-gated, default-off, host-side only: `STRATA_OPD_LOG=<path>` (tuples) + optional `STRATA_OPD_ROWS=<path>`
(break-position R rows). Logger = one buffered FILE*, one line per verify window; no sampler or verdict
semantics touched; the campaign's "env-gated, default off, one-line rollback" law (AGENTS.md campaign law 2)
and the STRATA_MTP_TAP precedent (mtp.cpp:68-93) are the pattern.

Attach points — all three verdict sites, same record shape:

1. **Serial lane** (the full chains; richest supervision):
   - window build generate.cpp:14403-14411 (`window[0]=x`, drafts at `window[1..T-1]`);
   - `ver.run(T, window, p, ...)` -> `outv[]` = the target's picks per row (:14421);
   - verdict scan :14449-14450: `while (a < T-1 && window[a+1] == outv[a]) ++a;`
   - commit `ver.commit(a+1)` :14471.
2. **Pipelined lane**: verdict :12956-12957 (`A.tok[a+1] == outp[a]`), commit :12958, the existing
   offered/accepted counters :12960-12961 (`draft_offered += A.T-1; draft_accepted += a`).
3. **Batch lane** (F-184 2-row slots; depth-1 tuples): the `finish` lambda generate.cpp:10310-10338 —
   draft `outb[i]` vs pick `w.tok[i+1]` (:10321), rescue branch :10324-10336 (log the rescue VERDICT with
   `rescued=1`; typical/ratio rescue is the only lossy row class, F-200 — flagged in the tuple so training
   can exclude or down-weight it), `commit_slot_prefixes(keep)` :10339.

Tuple format (replay.jsonl, one line per window):

```
OPD v1 doc=<id> band=<L0..L4> p=<window-start pos> T=<T> a=<accepted>
  ok=<i:draft:pick,...>          rows 0..a-1 that matched (window[i+1]==outv[i])
  rj=<a:draft:pick>              the break row: draft=window[a+1], pick=outv[a]; absent iff a==T-1
  resc=<0|1> lane=<ser|pip|bat>
```

Optional `opd-rows.bf16`: after a break, copy the break row's R from the verifier's device rows — the
`hash_rows` precedent does exactly this copy shape (`cudaMemcpy(hash_rows, ver.final_R_all(), ...)`,
generate.cpp:14431-14434) — convert bf16-RNE (the prefill.cpp:4204-4208 snippet), append, and reference it
from the tuple as a byte offset. ~0.22 rejections/window (T17C §5.5 accepted-mass arithmetic) x ~680k
windows ~= 150k rows ~= 3.1 GB.

Validation gate for the logger (before Pass A counts): one throwaway boot, 3 windows hand-checked against
the log's own acceptance counters (drafts accepted X of Y must equal the tuple sum), then the env is
deemed inert when unset (grep the boot log; the STRATA_MTP_TAP discipline).

### 4.3 Volume expectation

~1.29M generated tokens (ladder §3.2) at 180-200 t/s ~= 1.8-2.0 h + seed prefills (~6.3M tokens ~= 16 min)
+ parked-prefix follow-ups -> **Pass A ~= 2-2.5 h**. Tuples: ~680-790k windows -> ~680k lines (~90 MB) +
~150k rejections; replay.jsonl ~= 90-120 MB.

### 4.4 Explicitly OUT of Pass A v1 (deferred riders)

- Verifier p-list dump for the LK arm (trainer-spec §3.2.1 `plists.bin`): needs per-row top-64 of the
  target's post-chain distribution (device-side extraction; full-vocab copies would be 162 KB/row ~= 110 GB
  at corpus scale). Deferred: bounded design = top-64 at BREAK rows only (150k x ~0.5 KB), same attach
  points. If it lands, LK (trainer arm a4) is enabled; if not, a4 is skipped — the a1-a3/a5/a6 arms are
  unaffected. Decided at implementation time, not on the harvest critical path.
- LongSpec position-offset augmentation: trainer-side (`pos_aug` flag, trainer-spec §3.2.3); zero harvest cost.

---

## 5. PASS B — prefill-only hidden-state harvest

### 5.1 Which launcher/config

- Config COPY `custom-iq3s-harvest-8082.json`: byte-identical args to custom-iq3_s-8081.json:3-57 EXCEPT
  `"port": 8082` and `"log": ".../custom-iq3s-harvest.log"`. Every distribution-bearing arg stays: `--pack`
  iq3_s, `--native`/`--ple-gguf` shards, `--spec 6`, `--max-context 131072`, `--kv int8`, `--kv-resident
  32768`, `--spec-min-p 0.5`, `--prefill auto:32768`, `--mtp-q4 all`, `--mtp-hnorm stream`, `--pcie-frac
  0.35 --pcie-mode dma`, `--control-vector-scaled refusal-r.gguf:1.0` (serving includes the cvec — the
  harvest must too), `--mtp .../mtp/rt` (the drafter stays live: Pass B's per-request final window drafts
  exactly as served, and `on_chunk` feeds it).
- Launcher COPY `launch-custom-iq3s-harvest.sh`: standing exports verbatim + ONE added line:
  `export STRATA_PREFILL_DUMP_R_ALL=/home/svg/LLM/Custom-data/mtp-iq3s/r18-train/records/raw-r.bf16`
  (and `STRATA_OPD_LOG` only in the Pass A variant of the launcher). Boot pattern = prefill-probe.sh's
  `boot()`: pkill discipline, `bash campaign/prewarm.sh`, `nohup serve/server.py --engine strata --config
  ... --port 8082`, /health poll (campaign/prefill-probe.sh:12-40).
- Optional: `"--prompt-cache", "0"` in the harvest config ("0 = read every prompt from the start",
  generate.cpp usage text) for Pass B determinism. With one conversation per doc and no shared prefixes the
  cache never hits anyway; the flag only removes a variable.

### 5.2 What guarantees prefill-only

`"max_tokens": 1` per request — the campaign's measured prefill-only procedure ("max_tokens=1 -> wall ~=
prefill", prefill-probe.sh header; the probe's own samples() does exactly this). The engine prefills the
prompt in batched chunks (`n_batched = n_prompt - 1`, generate.cpp:13771 server-side equivalent :8653 block),
runs ONE final window for the single generated token, and stops. No decode loop, no meaningful drafting.
Requests are serial (one in flight), so rows land as contiguous per-request spans in the ledger.

### 5.3 Where the hook attaches (code, verified this session)

- `STRATA_PREFILL_DUMP_R_ALL` read per chunk inside the batched prompt walk:
  src/prefill/prefill.cpp:4193-4214. Per chunk: `cudaStreamSynchronize(m.cs)`, then in 512-row batches
  `cudaMemcpy` the chunk's final-R rows (`m.R + t0*D`, D = hc*n_embd = 10240), convert f32->bf16
  round-to-nearest-even in place (`u += 0x7fff + ((u>>16)&1)`), append rows in position order. Opens with
  "ab" per chunk — appends across chunks AND requests.
- The walk then calls the drafter with the same rows: `on_chunk(m.R, T, p0, err)` (prefill.cpp:4215-4226;
  callback type prefill.hpp:141). The server's lambda pairs `R_i` with `cur[p0+t+1]` ("cell i pairs R_i with
  the token at i + 1", generate.cpp:8653-8655) and feeds `draft_kv`/`mtp.prefill` (E-9, prefill.hpp:49-56) —
  i.e., the dumped rows ARE the drafter's prompt-side serving inputs, by construction.
- The one-shot twin of the same wiring: generate.cpp:13825-13831.
- Why final R is the right state: the drafter binds to the verifier's final rows at load (generate.cpp:14157)
  and consumes `ver.final_R_all()` rows per decode slot (:10320-10332, :10385-10394) and via
  `set_source_R` (:13013/:13042/:13086). The harvest stores the prompt-path instance of the same quantity.
- Diagnostics only (NOT the harvest path): `--dump-final-r` f32 records {int64 pos, int64 tok} + 10240 f32,
  written per TOKEN-loop position (generate.cpp:638, :2158, :13768-13769, :13969-13974); `STRATA_MTP_TAP`
  first-window drafter R_/sample_/head_logits_ for the trainer's G0 parity (mtp.cpp:68-93, :286-293,
  :1229-1240; F-86/F-87/F-88/F-91 capture laws already in the code).

### 5.4 Alignment ledger (the part that must not be fuzzy)

The dump is headerless; the ledger is the alignment artifact:
1. Driver renders + tokenizes each doc locally (same template file + same `strata_tokenizer` module, §3.2).
2. Per request: record t_send/t_recv, `usage.prompt_tokens` (P), then bytes written to raw-r.bf16
   (before/after offsets — requests are serial so spans are contiguous).
3. ASSERT `rows == P - 1` and `byte span == rows * 20480`. Any mismatch -> quarantine the doc, log it,
   continue; >= 1% quarantine rate aborts the pass (corpus criterion §3.3.2).
4. The packer (pure python, repo venv: numpy + the strata_tokenizer + gguf-py not needed) converts selected
   spans: bf16 rows -> f32, prepend `int64 pos (0-based doc-local), int64 tok_i` from corpus.jsonl ids ->
   records.f32 in the trainer-spec §3.1 format. Drop each doc's final row (its label is the generation
   header/EOS seam — the F-77 law).

### 5.5 Probe (30-45 min, before the full pass) — GO/NO-GO

1. Boot the harvest launcher; 5 docs (~100k tokens total) across bands.
2. Check: dump rate (expect a few % under the bare prefill rate 6,291-6,972 tok/s, T14B §2.1; F-236 leg
   measured 7,229 t/s at ~121k); row-count contract on all 5; finiteness of all rows; ledger arithmetic.
3. A/B the dump overhead: same docs with the env unset vs set, same boot class (the S25 dump adds one D2H +
   host bf16 conversion per chunk — both after a sync that on_chunk already performs, prefill.cpp:4215-4224;
   expected ~2-4%).
4. Numerics cross-check (informational, not a gate): the batched prefill's R rows are FP16-GEMM-path states;
   the drafter's decode-time inputs come from the verify path (q8_1 activations — the E-9 note documents the
   two paths' non-bit-identity, prefill.hpp:49-56). Quantify: one doc, two runs — run 1 with
   `--prefill-until <P>` (forces positions >= P through the token loop with `--dump-final-r` armed), run 2
   normally (S25 covers position P) — compare the same position's row cos. Expect > 0.999; record in PROBE.md
   so the trainer knows the residual-path jitter bound. This matters because the drafter SERVES both regimes
   (first window = prompt-path state; chain windows = verify-path state).
5. GO: proceed to full Pass B. NO-GO: if rows != P-1 (measure the true constant — if the walk covers all P
   positions, adopt the measured constant and drop the final row at pack time), or rate < 5k tok/s
   (investigate before burning the pass).

---

## 6. Storage math (T11.R3's 160-320 GB estimate, confirmed and refined)

Per-row size: `R = hc x n_embd = 4 x 2560 = 10,240` values.

| Artifact | Per-unit | v1 (8.0M rows) | 16M expansion |
|---|---|---|---|
| raw-r.bf16 (S25 rows) | 20,480 B/row | 163.8 GB (152.6 GiB) | 327.7 GB (305.2 GiB) |
| records.f32 (packed) | 40,976 B/row | 327.7 GB — SEE BELOW | |
| token ids in corpus.jsonl | 4 B/row | 32 MB | 64 MB |
| ledger + manifest | — | < 10 MB | < 10 MB |
| opd/replay.jsonl | ~130 B/window | ~90-120 MB | — |
| opd/opd-rows.bf16 | 20,480 B/rejection | ~3.1 GB (150k) | — |

- **T11.R3's "160-320 GB bf16" is CONFIRMED** as the 8-16M-position envelope (163.8 / 327.7 GB decimal).
- **Refinement vs "our 48-layer geometry": the dump size is INDEPENDENT of n_layers = 48.** The MTP input is
  the FINAL residual only (one 10,240-value row per position, generate.cpp:14157 bind + the S25 rows). A
  per-layer dump (`--dump-layers`, generate.cpp:497/:862, `n_layers * hc * n_embd` f32/position) would be
  1.97 MB/position ~= 15.8 TB at 8M — infeasible and unnecessary (T14B §2.2: "overkill for MTP; the head
  reads only the last R"). The harvest's cost driver is POSITIONS, not layers.
- **f32 is rejected, bf16 only.** `--dump-final-r`'s 40,976 B f32 records at 8M = 328 GB and would DOUBLE
  via the packed copy; S25's bf16 at 20,480 B is what the trainer wants anyway (trainer-spec §3.4 "bf16 sidecar
  conversion at write"). To avoid holding raw AND packed simultaneously: pack IN SLICES (process raw-r.bf16
  per doc span, append records.f32, optionally shrink raw-r.bf16 with `ftruncate` per confirmed span) — peak
  disk ~= 330 GB for both formats coexisting briefly, inside the **705 GB free** on /home (df, 2026-10-18;
  was 735 GB at T14B time — the same downward drift trainer-spec §3.4 already priced).
- Deletion law (trainer-spec §8): raw-r.bf16 + records.f32 are deleted after the winning arm's export passes
  the audit; opd/ and corpus/ (a few hundred MB) persist as the reproducible data provenance.

## 7. Runtime estimate (ONE 5090 + 9950X3D, exclusive)

| Stage | Estimate | Basis |
|---|---|---|
| Harvest boot (x2: Pass A, Pass B) | 5-10 min each | 85 GB pack load + prewarm (battery-boot pattern) |
| Corpus assembly + render + tokenize (driver, CPU) | ~30-45 min | ~640 docs, ~8M tokens of jinja render + BPE; 9950X3D |
| OPD logger build + validation boot | 0.5 day (the one eng item) | §4.2 |
| Probe (§5.5) | 30-45 min | 100k tokens + 2-run cross-check |
| PASS A | 2-2.5 h | 1.29M generated tokens @ 180-200 t/s + 6.3M seed prefill @ ~6.5k t/s + parked follow-ups |
| PASS B | 45-60 min | 8M tokens @ 6.3-7.2k t/s ~= 19-21 min pure prefill + 640 requests x 1-2 s overhead + 2-4% dump tax |
| Packer + validation | ~1 h | bf16->f32 conversion is bandwidth-trivial (164 GB at NVMe/DRAM speed); finiteness scan the same |
| **Total compute** | **~4-5 h** (one overnight slot with margin) | |

Anchors: T11.R3's "~2-6 h harvest" envelope CONFIRMED (its estimate assumed the corpus existed; our Pass A
adds the self-distill generation and still fits). T14B P1's "45 min (batched) / 22 h (per-token)" applies to
Pass B alone; the per-token fallback is DEAD (native packs force the batched path anyway, §1.2). 16M
expansion adds ~1 h of harvest.

## 8. Preconditions checklist (GO/NO-GO before any boot)

1. **PACK RE-QUANT FIRST (the sequencing law).** T17 CONSOLIDATED VERDICT item 1: "PACK RE-QUANT ... BEFORE
   the T14A harvest"; T17F §6: "land BEFORE the T14A hidden-state harvest so the drafter trains on the
   serving distribution"; T17F §5 open question 1 asks exactly this question and answers it with the
   sequence. If the re-quant (T17F #1: +6-10% M1, 1-2 CPU-days, one-boot falsifier) has NOT landed and the
   harvest must proceed, the harvest is stamped PRE-REQUANT in manifest.json and a FULL re-harvest (Pass A +
   Pass B, ~4-5 h) is planned on re-quant GO — the R rows AND the OPD tuples are pack-coupled (the pack's
   prefill bits shift the hiddens themselves; F-230's coupling mechanism).
2. Box idle; batteries serial (T16 preamble law). The standing 8081 server is stopped for the harvest boots
   (prefill-probe.sh pkill discipline) or the harvest runs in a verified-idle window; never boot-adjacent to
   a running battery (T14A gates law).
3. Disk: >= 200 GB free on /home at Pass B start (705 GB now; trip-wire check per 100 docs).
4. Corpus assembled and acceptance-checked (§3.3) BEFORE Pass B; corpus.jsonl is the single source of
   token ids.
5. OPD logger validated (§4.2 gate) before Pass A starts; env verified inert when unset.
6. Probe GO (§5.5) before the full Pass B.
7. No trainer venv / torch needed for the harvest (engine-side + repo-venv python only). P0 of the trainer
   spec is independent and may proceed in parallel on the CPU.

## 9. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Alignment drift (driver render/tokenize != engine): the F-104-class silent killer | Same template file + same strata_tokenizer module; per-request length assertion quarantines ANY drift (exact-length check, not heuristic); >= 99% pass-rate gate; PROBE.md records the constant |
| R2 | Dump in a slow path (T14B risk 3) | MOOT for the mechanism (S25 is the batched path, §1.1); the probe still measures the tax (~2-4% expected) and enforces >= 5k tok/s |
| R3 | Pack re-quant lands after the harvest | Precondition §8.1: sequence it first, or pack-stamp + planned re-harvest (~4-5 h) |
| R4 | No MIMIC corpus; domain mismatch of substitutes | Mixed plan: repo prose (content-matched, F-236-promoted) + HQ diversifier (A/B-gated) + >= 50% self-distilled labels; the trainer's G0.2/M3b gates catch under-training; RH's warning that domain-narrow data underperforms is heeded by the mix (T14A R3) |
| R5 | Seed-pool reuse (675k tokens drawn ~9x) inflates seed memorization | Rotation makes (content, position) pairs distinct (F-105 position law); labels that matter are the self-distilled replies; optional HQ expansion; dedup on (content, position-range) |
| R6 | Deep-band generation quality (the model's own deep outputs are weaker — acceptance 53.1%) | That is the point: harvest the SERVED distribution, weakness included; Pass A tuples at depth are precisely the deficit supervision; L0's 30% protects pos-0 |
| R7 | Context overflow on L4 docs | Hard cap: rendered doc <= 126k < 131,072 (§3.2); driver enforces pre-send |
| R8 | Partial writes / server death mid-pass | Append-only dump + serial requests = per-doc byte spans; ledger marks the last complete doc; resume = continue from the ledger tail; quarantined spans skipped at pack |
| R9 | OPD logger perturbs the hot path | Buffered fprintf (~130 B/window), env-gated default-off, inertness verified per §4.2; never armed in a measured battery boot |
| R10 | Disk fill mid-pass | Trip-wire free-space check per 100 docs; 200 GB floor (§8.3); artifact deletion law post-export |
| R11 | The rescue rows pollute OPD supervision (typical/ratio rescue is lossy, F-200) | `resc=1` flag on batch-lane tuples; trainer down-weight or excludes (trainer-spec owns the flag's consumption) |
| R12 | Batched-path vs verify-path R numerics differ (E-9 non-bit-identity) | Bounded by the §5.5.4 cross-check (cos > 0.999 expected); the drafter already serves both regimes; recorded in PROBE.md for the trainer |

## 10. Operator runbook (step by step)

```
 0. PRECONDITIONS — walk §8 item by item; resolve the re-quant sequencing question explicitly
    (either it landed, or the manifest will say PRE-REQUANT and a re-harvest is scheduled).
 1. ASSEMBLE CORPUS (CPU only, repo venv):
      python campaign/t14b_harvest_driver.py assemble      # stage 1: build docs per §3.2 from the seed
                                                           # pools, rotate slices, render jinja, tokenize,
                                                           # dedup, write corpus/corpus.jsonl + plan
    Check §3.3 criteria 5-6 inputs (lengths, dedup, provenance).
 2. BUILD OPD LOGGER (the half-day eng item, §4.2): env-gated STRATA_OPD_LOG/STRATA_OPD_ROWS at the three
    verdict sites (generate.cpp:14449-14450, :12956-12957, :10310-10338; R-row copy per :14431-14434).
    Validate: one throwaway boot, 3 windows hand-checked vs the DONE-line counters, inertness check.
    NOTE: this is the only engine-code change in the whole harvest; everything else is env + driver.
 3. PASS A:
      cp custom-iq3_s-8081.json custom-iq3s-harvest-8082.json   # edit port->8082, log path
      cp launch-custom-iq3s-8081.sh launch-custom-iq3s-harvest.sh # add STRATA_OPD_LOG/STRATA_OPD_ROWS
      bash launch-custom-iq3s-harvest.sh                        # /health on 8082
      python campaign/t14b_harvest_driver.py distill            # one conversation per doc-seed, temp 0.6
                                                                # top-p 0.95, one in-flight; deep docs get
                                                                # parked-prefix follow-ups; parse DONE lines
                                                                # into opd/acceptance-by-band.txt
    Assemble final corpus docs (user seed+instruction, assistant reply) into corpus.jsonl; re-check §3.3.
 4. PASS B:
      cp launch-custom-iq3s-harvest.sh launch-custom-iq3s-harvestB.sh  # swap OPD env for:
      #   export STRATA_PREFILL_DUMP_R_ALL=/home/svg/LLM/Custom-data/mtp-iq3s/r18-train/records/raw-r.bf16
      pkill per prefill-probe.sh discipline; bash campaign/prewarm.sh; boot; /health.
      python campaign/t14b_harvest_driver.py harvest            # replay every doc as a FRESH conversation,
                                                                # max_tokens=1, serial; write ledger.jsonl
                                                                # spans; assert rows == prompt_tokens-1;
                                                                # free-space trip-wire per 100 docs
 5. POST-VALIDATE: finiteness scan; ledger vs file size; band histogram -> manifest.json (pack sha256s,
    config hash, env hash, dates, boot commands, PRE/POST-REQUANT stamp).
 6. PACK: python campaign/t14b_harvest_driver.py pack          # select band mix per §3.2 ladder, convert
                                                                # bf16->f32, prepend (pos,tok) int64s,
                                                                # drop final row per doc -> records.f32
                                                                # (trainer-spec §3.1 format); write PROBE.md
 7. TEARDOWN: stop the harvest server; verify the standing config/launcher untouched (git status CLEAN of
    them — copies only); leave corpus/ opd/ manifest + an F-entry with the numbers (campaign law 6:
    record everything, results into campaign/results/).
 8. HANDOFF: trainer-spec §3/§8 consumes records/ + opd/; gates-spec owns everything after G0.
```

Driver scripts (`t14b_harvest_driver.py`) are NEW pure-python tooling under campaign/ (no engine code);
their only imports are stdlib + numpy + jinja2 + strata_tokenizer from the repo venv.

## 11. Contradictions and their resolutions

1. **"The harvest pass should ALSO log rejection/pick pairs" (task framing, riding one pass) vs prefill-only
   = no decode = no rejections.** Resolved as TWO passes over ONE corpus (§4/§5): Pass A decodes (rejections
   exist), Pass B prefills (rows exist). T17F §3.1's own wording ("during the MIMIC-corpus harvest ... ALSO
   run the drafter") already implied the drafter must RUN — i.e., decode — for rejection tuples; the split
   makes that explicit and keeps each pass at its natural throughput.
2. **`--dump-final-r` as the harvest mechanism (trainer-spec §3.1, T14B P1) vs the native-pack constraint.**
   On our IQ3_S pack the one-shot generate mode REQUIRES `--prefill CHUNK` (generate.cpp:3117-3123), so the
   per-token hook sees only the last prompt position + generated tokens. Resolved: S25
   (`STRATA_PREFILL_DUMP_R_ALL`) is the mechanism; `--dump-final-r`'s record format is preserved as the
   PACKER'S OUTPUT format (trainer contract intact) and the hook for the numerics cross-check probe (§5.5.4).
3. **T14B's "~50-line batched dump extension" (risk 3 / P1) vs the current tree.** The extension SHIPS:
   prefill.cpp:4193-4214. The remaining new code is the OPD logger only (~half day), matching the task's own
   estimate. Trainer-spec §3.3's per-token-vs-batched probe question is hereby answered: batched, zero build.
4. **Trainer-spec §3.1 "16M positions target" vs this spec's 8.0M v1.** Not a conflict: RH's 5k-sample
   precedent says scale is not binding (T17C §5.1, T17F item 6: +1-3pp for 4x data), the trainer's G0.2 gate
   catches under-training, and 16M remains a one-command expansion (the ladder doubles per band; +~1 h
   harvest, +164 GB). 8M v1 buys the gates a first read sooner.
5. **Trainer-spec "oversample deep >= 2x traffic share" vs this spec's ~7-30x (35% of rows at 64k+).** The
   F-236 finding post-dates nothing — both spec teams read it — but the -40.5pt primary-deficit result plus
   the parked 262k switch justify exceeding the floor; L0's 30% share protects pos-0, and pack-time remixing
   (§3.2) makes the choice reversible without re-harvesting. Recorded as a deliberate divergence for the
   G0.2 read to adjudicate.
6. **"MIMIC corpus" (T11.R3/T14A/T17F language) vs reality.** No such corpus exists anywhere reachable
   (§3.1); the name appears to have been a placeholder that congealed into an assumption. F-235/F-236 already
   banked the absence; this spec re-verified Custom-data AND the mount table and demotes /mnt/DATA itself to
   conditional (unmounted, 2026-10-18). All downstream docs should say "the T14B harvest corpus" and cite
   §3.1.

## 12. Open questions

1. Does the server prompt walk cover exactly [0, P-1) positions (rows == P-1)? The probe measures the
   constant; if rows == P, drop the final row at pack time (§5.5.5). Cheap, decisive, first thing measured.
2. Does the HQ diversifier earn its <= 25% share? One-doc A/B at matched depth: acceptance of an HQ-seeded
   reply vs repo-prose-seeded reply, same band (parse from Pass A's own DONE lines). If HQ wins or ties, keep;
   if it loses, repo-prose + self-distill only.
3. Should Pass A also log the drafter's q top-64 at break rows (cheap; complements the deferred target
   p-list, §4.4)? Decided at logger implementation; both feed trainer arms a4/a5.
4. Does the rescue flag (R11) belong in the CE upweight at all, or only `rj` rows from serial-lane windows?
   Trainer-spec's replay_weight config owns it; the tuple carries the flag either way.
5. 262k ladder arm: deferred to W3 (§3.2). If the switch GOes before a re-harvest, add one L5 band
   (126k-245k) at ~10% — the driver's band table is data, not code.

## 13. Sources

Code (read this session; engine = /home/svg/LLM/strata-custom-2):
- src/prefill/prefill.cpp:4180-4226 — STRATA_PREFILL_DUMP_R (every 64th, debug), STRATA_PREFILL_DUMP_R_ALL
  (S25, bf16-RNE, [n][hc*n_embd], append across chunks/requests), on_chunk invocation + sync (:4215-4226).
- include/strata/prefill/prefill.hpp:49-56 (draft_kv E-9 contract incl. the FP16-vs-Q8_1 non-bit-identity
  note), :141 (on_chunk callback type).
- src/program/generate.cpp:638/:2158 (--dump-final-r parse), :13768-13769 (open), :13969-13974 (per-token
  record write, int64 posrec[2] + hc*n_embd f32); :13771-13883 (batched prompt walk, n_batched, on_chunk
  wiring :13825-13831, next-token pairing :13828); :8653-8689 (server-side on_chunk: pairing :8655, draft_kv
  :8663, PP progress); :14157 (mtp.bind on ver.final_R_all); :10320-10332/:10385-10394 (batch handoff);
  :13013/:13042/:13086 (set_source_R); :14403-14411 (window build), :14421 (ver.run -> outv), :14449-14450
  (serial verdict), :14471 (commit), :14431-14440 (final_R_all D2H copy precedent); :12956-12961 (pipelined
  verdict + counters); :10310-10339 (batch finish: outb[i] vs w.tok[i+1] :10321, rescue :10324-10336,
  commit_slot_prefixes); :13518/:13599 (DONE line format), :13364 (avg-T line); :10685 (slot accept line);
  :3117-3123 (native pack requires --prefill CHUNK); :616/:2126-2146/:2157 (--prefill / --prefill-until);
  usage text --prompt-cache ("0 = read every prompt from the start"), --max-new default 16 (:858 region).
- src/core/mtp.cpp:68-93 (STRATA_MTP_TAP dump), :286-293 (pinned alloc at bind, F-91), :1229-1240 (in-capture
  copies), :690-781 (bind, 40,525-row head gather).
- serve/server.py:3786-3798 (render_prompt, jinja), :6703-6717 (strata_tokenizer load from the pack
  tokenizer dir), :141 (template effort note).
- launch-custom-iq3s-8081.sh:8-42 (standing env arms + Launch Law exec line); custom-iq3_s-8081.json:3-72
  (standing args: --pack/--native/--ple-gguf, --spec 6, --max-context 131072, --kv int8, --prefill auto:32768,
  --mtp-q4 all, --mtp-hnorm stream, cvec refusal-r.gguf, --mtp rt; parallel 6; draft_vocab en).
- campaign/prefill-probe.sh (max_tokens=1 prefill-only procedure; boot/pkill/prewarm pattern);
  campaign/depth-lane-real.py:30-47 (SOURCES pool, 3.27 chars/token calibration), :49-63 (request shape),
  :76-84 (acceptance-line parse); campaign/mimicbench2.py (the "mimic" benchmark, not a corpus).
- Tools: tools/mtp_fetch.py / mtp_pack.py / mtp_rt.py (export chain context; not exercised by the harvest).
- Rig/filesystem (verified 2026-10-18): df (/home 952 GB, 705 GB free), ls/du ~/LLM/Custom-data (no MIMIC),
  /mnt mount table (DATA unmounted; HQ/STORE/SCRATCH ntfs3), /mnt/HQ/store listing (KnowledgeBase/DATASET
  4.5 GB, Repo Collection), find for *mimic*/*corpus*/*pairs-r17* under /home/svg/LLM and .zcode (absent).

Banked campaign and dossiers:
- SVG-CUSTOM-TODO.md: T11.R3 :793-806; T17 QUEUE :1096-1236; T17 CONSOLIDATED VERDICT :1215-1236 (re-quant
  before harvest :1224); STEP 0 RESULTS :1238-1269 (depth-split :1250-1255).
- SVG-CUSTOM.md: F-235 :8074-8102; F-236 :8104-8137 (position -40.5pt primary / content -14.9pt).
- campaign/T14B-specs/trainer-spec.md (§3 record contract, §8 artifact layout, §3.2 OPD/p-list/aug riders).
- campaign/T14B-specs/gates-spec.md (downstream gate owner).
- campaign/research/T14A-fastmtp-method-recipe.md (F1/F2 data laws; Arm 0 items 1-3; R1-R5).
- campaign/research/T14B-fastmtp-training-infra.md (P1, §4 budgets, §6 risks, F-77..F-118 laws).
- campaign/research/T17C-spec-algo-leaps.md (§2 contract map; §5.1; rejection mass ~0.22/window).
- campaign/research/T17F-training-quant-context.md (§3.1 Draft-OPD; §3.4 LongSpec/MagicDec; §6 sequencing).
- AGENTS.md (campaign law: env-gated default-off, record everything, serial batteries, Launch Law).

arXiv IDs (banked, verified in the dossiers; no new research): FastMTP 2509.18362; Draft-OPD 2605.29343;
LongSpec 2502.17421; VAT 2608.30135; LK losses 2602.23881; AdaSPEC 2510.19779; Red Hat Speculators 0.6.0
article (developers.redhat.com, 2026-09-08; 5k samples / 443 s on 2xH200).

## 14. Methods note (what did NOT work / traps for the next reader)

- DO NOT plan the harvest on `--dump-final-r` alone: native packs force the batched prefill path, so the
  per-token hook yields ~1 record per request (found by reading the native-pack gate at generate.cpp:3117,
  not by burning a boot).
- DO NOT size the dump from n_layers: the harvest is one final-R row per position (10,240 values); the
  48-layer number only enters via the rejected --dump-layers option (15.8 TB class).
- /mnt/DATA is in the task framing as "read-allowed frozen copies" but is NOT MOUNTED on this box
  (2026-10-18) — any plan that depends on it is dead until the operator mounts it; /mnt/HQ is the live
  read-allowed volume.
- The "MIMIC corpus" name: searching it finds only benchmark result files (mimicbench2.py lineage) — there
  is no corpus behind the name anywhere on the box; do not re-search.
- The dump files open in "ab" mode per chunk with no magic/headers: any tooling that assumes a self-describing
  format will misread; the ledger is mandatory (§5.4).
- Unproductive: no web fetches were needed (task: "no new research required"); all external claims ride the
  dossiers' verified arXiv IDs. The one repo-venv uncertainty left open is cp-related and belongs to the
  trainer spec (its §2.2 wheel rule), not the harvest.
