# T14B-exec-A — P1 teacher-data harvest: the execution-ready spec (dump lane, batched extension, corpus plan, probe runbook)

Date: 2026-10-18. Leg A of the relaunched T14B trio (B = module/G0 parity, C = trainer/export, both filed).
Scope: the P1 phase of the MTP drafter retrain — harvest the teacher signal `(pos, token, final-R)` from the
FROZEN IQ3_S target by replaying a seeded corpus through a patched OFFLINE engine, at batched-prefill speed.
READ-ONLY audit for this session: nothing built, nothing booted, the 8081 lane and `/mnt/DATA` untouched.
Every line number below was read at the current HEAD (post-F-236, `engine/strata` built Oct 17 11:15); T14B's
cited dump lines (13683–13688) and exec-C's are STALE — the file grew ~200 lines since 10-16.

---

## 0. Summary

The harvest lane is **CLI generate, one boot per corpus stream, one stream per tokens-file** — not serve.
Two facts discovered this session reshape T14B's P1 sketch:

1. **The existing `--dump-final-r` hook is DEAD CODE for our target.** It lives in the per-token loop, but a
   native pack (ours) breaks out of that loop at generate.cpp:13892 before a single record is written —
   "a native pack's last prompt token is the first verify window", and everything decode-side runs verify
   windows. Running today's flag unpatched yields an empty file. This corrects T14B §2.2/§3-P1.
2. **The batched extension is smaller than T14B estimated** because the prefill path already carries the exact
   buffer to a user callback: `Prefill::on_chunk(R_rows, T, p0)` hands over the chunk's final multi-stream
   residual rows (device, T×10240) with the stream synchronized (prefill.cpp:4215-4226). The patch is ~35
   lines in generate.cpp: install a chaining lambda after the existing MTP `on_chunk` block, plus a ~10-line
   record write in the verify-window loop for the last prompt position. No core-lib changes, one rebuild
   target (`strata`).

**Record format (VERIFIED at source — supersedes T14B §2.2 and exec-C §2.1):** `{int64 pos, int64 tok}` +
`float32[4×2560]` row = **40,976 B/record** (generate.cpp:13969-13973: `const int64_t posrec[2] = {pos, tok};`
then `final_r_host`, `g.hc × g.n_embd` floats). Exec-C's pinned 40,968 B / int32-int32 contract is off by 8
bytes per record and would misalign every read after the first; §2.1 of exec-C needs a dated revision (see
§7.1 and §9).

Corpus (decided, §4): ~180 streams, all read from position 0, lengths 131,072 / 65,536 / 24,576 (40/60/80),
≈ 11.0 M positions ≈ 451 GB f32 — deep positions (≥96 k) oversampled to ~13 % of corpus per F-235/F-236;
content = mimic-lane rendered conversations + repo-prose study turns + seeded ladders; response positions are
harvested at prefill speed by embedding assistant turns IN the streams. Probe first (§5); budget table §6.

---

## 1. THE DUMP LANE — `--dump-final-r` end to end (read at current HEAD)

All references: `src/program/generate.cpp` unless noted.

**Option surface**
- :638 `std::string dump_final_r;` — with the doc comment ":636-637 *append the residual after the last layer
  (hc x n_embd floats, the MTP draft head's input) to this file. **Token path only.**" The "token path only"
  caveat is load-bearing (below).
- :2158 `else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");`
- Usage text :859-861 lists the sibling dumps (`--dump-logits`, `--dump-residual`, `--dump-layers`).

**Open + buffer**
- :13768 `std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");`
  — opened ONCE, mode `"wb"` (truncate), no error message if fopen fails (silently inert).
- :13769 `std::vector<float> final_r_host(final_r ? (size_t)(g.hc * g.n_embd) : 0);` — the D2H staging row.

**Where it sits in the control flow (the decisive fact)**
- The `--serve` request loop ENDS at :13753 (`return 0;` inside `if (o.serve)`, block from :7332). Everything
  below — the fopen at :13768, the batched prefill at :13772-13883, the per-token loop at :13890+ — is
  **CLI-generate-only**. Under `--serve` the flag parses and is never used. There is no serve-mode dump lane.
- :13771 `n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1`
  — the batched prompt path covers positions `[0, n_prompt-1)` when `--prefill` is on.
- :13890-13892 `for (int64_t pos = pos_start;; ++pos) { if (native_pack) { spec_pos = pos; break; } ...` —
  **for a native pack the per-token loop exits immediately.** `native_pack` = `expert_layout().native`
  (:2926), true for our `--pack iq3_s --native <shard1>` boot (production logs show verify-window decode:
  "avg T … tokens/window"; the doc at :6067 says "a native pack runs verify windows only"). Decode and the
  last prompt position run the verify-window loop instead (:14358+, `ver.run` at :14421).
- Therefore today's hook writes records ONLY when a non-native pack runs tokens through the per-token loop —
  never for the IQ3_S target. T14B's "the hook lives in the per-token prompt loop; probe decides" premise
  resolves to: the probe would measure 0 records. The extension (§2) is mandatory, not optional.

**The per-token record write (the format contract)**
- :13969-13973, inside the per-token loop after the session step:
  ```cpp
  if (final_r != nullptr) {
      cudaMemcpy(final_r_host.data(), ss.R, final_r_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
      const int64_t posrec[2] = {pos, tok};
      std::fwrite(posrec, sizeof posrec, 1, final_r);
      std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);
  }
  ```
- **Record layout (little-endian, raw memory, no padding, no file header):**
  `int64 pos` (absolute position), `int64 tok` (the token id whose forward produced this row),
  `float32[4][2560]` row-major (4 hyper-connection streams × 2560; `R` is `(hc, n_embd)`, layer.hpp:453-456;
  `g.hc=4, g.n_embd=2560` — pre_fc_norm_hidden 10240 and block_inject 4×10240 in mtp-inventory.md).
  **40,976 bytes per record.** `pos` and `tok` are the values at the loop iteration (`ss.R` is the residual
  AFTER processing `tok` at `pos`).
- **Semantics for the trainer (T14B §3-P1 alignment, confirmed):** record i holds `(R_i, tok_i)`; the
  drafter's serving input at prediction point i is `(R_i, emb(tok_i))` and its label is `tok_{i+1}` — pure
  offset bookkeeping along the stream. Serving proof: the drafter is bound to `ver.final_R_all()` /
  `ss.R` rows exactly like this (bind :14157; handoff drafts at :10385/:13568 read
  `final_R_all() + t*stride`).
- **What `tok` is at each position:** the INPUT token at that position (prompt token id, or the sampled id
  for generated positions) — :13908/:13921 (`ss.ple_token = tok; put_input(tok, pos)`).
- Working state today (unpatched): CLI + non-native pack only; empty file for us. The S25 hook (§2.3) is the
  only batched dump that works TODAY.

**Existing siblings worth knowing**
- `--dump-layers PATH` (needs `--no-capture`), `--dump-logits`, `--dump-residual`, `--window-hashes`
  (:14431-14441 — already does a T×10240 D2H of `ver.final_R_all()` per window: the window-side copy pattern
  the patch reuses).
- **`STRATA_PREFILL_DUMP_R_ALL`** (prefill.cpp:4193-4214, "S25"): env-gated, **works today**, writes every
  batched-prefill position's final row as **bf16 (RNE)** — headerless `[n][hc*n_embd]`, appended in position
  order across chunks. It is the zero-code fallback lane (§7.6) and the output-side precedent for bf16
  conversion. Caveats: no pos/tok header, misses the last prompt position (window path), and its host-side
  f32→bf16 conversion costs ~1-2 s per 32 k-chunk on this box's scalar path — fine for fallback, not the
  primary lane.

---

## 2. THE BATCHED EXTENSION — patch-level spec (~35 lines, one file)

**Design decisions (decided here):**
- Hook the PROGRAM layer (generate.cpp), not prefill.cpp: the lambda needs `o.tokens` for the tok column, and
  prefill.cpp already exposes everything via `on_chunk` (prefill.cpp:4223 `on_chunk(m.R, T, p0, err)`, called
  after `cudaStreamSynchronize(m.cs)` at :4217 — the rows are settled device memory, T×10240, valid until the
  next chunk per prefill.hpp:138-141).
- Record format IDENTICAL to the per-token hook (i64+i64+f32[10240], 40,976 B) — one reader for both lanes.
- Env-gated **default off** per campaign law (AGENTS.md, Law 2): both `--dump-final-r` (already opt-in) AND
  new env `STRATA_DUMP_FINAL_R_BATCH=1` must be set for the batched/window writes to activate. Production
  behavior unchanged without both.

**Insertion point 1 — generate.cpp, immediately AFTER the MTP `on_chunk` block (after :13832's closing
brace, BEFORE :13833 `if (!kvg_started)`).** At this point `prefill.on_chunk` holds the draft_kv lambda iff
`--mtp` was passed (set at :13825-13831); the harvest boot passes no `--mtp`, so it is empty — chain anyway
so a `--mtp` harvest stays correct.

```cpp
// T14B-exec-A: harvest the teacher signal from the batched prompt path (env-gated, default off).
// Record = {int64 pos, int64 tok, float32[hc*n_embd]} — byte-identical to the per-token hook's layout.
if (final_r != nullptr && std::getenv("STRATA_DUMP_FINAL_R_BATCH") != nullptr) {
    std::fprintf(stderr, "strata generate: batched final-R dump armed (%lld positions expected)\n",
                 (long long) n_batched);
    const auto prev = prefill.on_chunk;                       // chain the MTP draft_kv lambda if present
    const int64_t HCN = g.hc * g.n_embd;
    std::vector<float> rows;                                  // grows to the first chunk's T*HCN, then reused
    prefill.on_chunk = [&, prev](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
        rows.resize((size_t) T * (size_t) HCN);
        if (cudaMemcpy(rows.data(), R_rows, rows.size() * sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess) { e = "final-R dump D2H failed"; return false; }
        for (int64_t t = 0; t < T; ++t) {
            const int64_t posrec[2] = {p0 + t, o.tokens[(size_t) (p0 + t)]};
            std::fwrite(posrec, sizeof posrec, 1, final_r);
            std::fwrite(rows.data() + (size_t) t * (size_t) HCN, sizeof(float), (size_t) HCN, final_r);
        }
        return prev ? prev(R_rows, T, p0, e) : true;
    };
}
```
Scoping verified: `final_r` (:13768), `g`, `o`, `prefill`, `n_batched` (:13771) are all in scope here; the
lambda runs synchronously inside `prefill.run` (:13839) on the caller thread (single-GPU harvest — see §7.5),
so by-ref captures outlive every call. `o.tokens[p0+t]` is valid: `run()` guarantees `p0+T ≤ n_batched`.

**Insertion point 2 — generate.cpp verify-window loop, after the window's error check (between :14430 and
:14431 `if (hashes != nullptr)`) — covers position n_prompt-1 (first window, T=1, `window[0] = x = tok` at
:14314-14315/:14403, `p = spec_pos`):**
```cpp
// T14B-exec-A: harvest the verify-window rows too (the last prompt position; decode positions with --max-new > 1).
if (final_r != nullptr && std::getenv("STRATA_DUMP_FINAL_R_BATCH") != nullptr) {
    const int64_t HCN = g.hc * g.n_embd;
    wr_rows.resize((size_t) T * (size_t) HCN);                // std::vector<float> wr_rows; declared near hash_rows (:14334)
    if (cudaMemcpy(wr_rows.data(), ver.final_R_all(), wr_rows.size() * sizeof(float), cudaMemcpyDeviceToHost)
        != cudaSuccess) { std::fprintf(stderr, "strata generate: window final-R dump D2H failed\n"); return 1; }
    for (int i = 0; i < T; ++i) {
        const int64_t posrec[2] = {p + i, window[(size_t) i]};
        std::fwrite(posrec, sizeof posrec, 1, final_r);
        std::fwrite(wr_rows.data() + (size_t) i * (size_t) HCN, sizeof(float), (size_t) HCN, final_r);
    }
}
```
With the runbook's `--max-new 1` exactly ONE window runs (`while produced < max_new`, :14358;
`first_window forces T=1`, :14365), so the window lane contributes exactly one record per stream — position
`n_prompt-1` — and no draft-garbage rows (drafts only fill `window[1..]` in later windows).

**Insertion point 0 (one line at :13768-13769, after fopen):** a big stdio buffer and a loud failure —
```cpp
if (final_r == nullptr && !o.dump_final_r.empty()) { std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_final_r.c_str()); return 1; }
if (final_r != nullptr) std::setvbuf(final_r, nullptr, _IOFBF, 1 << 26);   // 64 MiB: 40 KB record writes otherwise thrash the 8 KB default buffer
```

**Build + gate:** `cmake --build build --target strata -j16` (target `strata` links generate.cpp —
build/build.ninja:2248). Gates in §5/§7. Expected cost of the patch on the measured prefill: one 1.34 GB
pinned-capable D2H + one 1.34 GB page-cache fwrite per 32 k-chunk (~5 s of compute) ≈ 5-8 % wall.

**Coverage invariant per stream (with `--max-new 1`):** records == n_prompt == tokens-file length
(`[0, n_prompt-1)` from on_chunk + 1 from the first window). This is the primary machine check (§7.2).

**Not in v1** (exec-C §3.3 wants it for `--replay-rejection` v2): per-position (draft, accepted?, target-pick)
telemetry. The insertion points above are where it would ride; do not build it now.

---

## 3. (merged into §2/§4) — no separate lane

The dump lane and the extension are one lane: CLI generate, patched, env-gated. There is no serve-mode harvest
(§1, §7.3).

---

## 4. THE CORPUS PLAN (decided)

**Goal:** ~11 M teacher records covering positions 0..131,072 (today's production `--max-context`), with
deep positions OVERSAMPLED — F-235/F-236 (SVG-CUSTOM.md:8089-8093, :8104+; F-236 commit 315919bc): fresh-read
deep acceptance 53.1 % (635/1195 at ~123 k, real corpus) vs 93.6 % on the KV-reuse continuation = **-40.5 pt
position-primary deficit**, filler-vs-real at matched depth = -14.9 pt content-secondary. The drafter the
retrain produces must be good exactly where the current one collapses: deep positions on fresh reads.

**Why prompt-prefill-only harvesting covers response positions too (decided):** the drafter's input is
`(R, tok)` — invariant to whether `tok` arrived via decode or prompt. Render multi-turn conversations
(user turns AND assistant turns) into the streams; every position, including what would be response spans,
is then harvested at 6-7 k tok/s prefill speed. Decode-class harvesting is not merely slow — for a native
pack the per-token dump does not exist (§1); the window path at decode speed (~160-240 tok/s) would be 17-22 h
for the same coverage and adds sampling nondeterminism. `--max-new 1` keeps a deterministic tail.

**Streams = boots.** One CLI run ingests one `--tokens-file` (pretokenized ids; usage :780-783). Model load
per boot is the overhead floor, so the ladder uses FEW, LONG streams — every stream reads from position 0,
so shallow positions get redundancy from ALL streams (180× at 0-4 k), and deep bands come from the 131 k
tails:

| Ladder rung | streams | tokens each | positions | deep-band contribution |
|---|---:|---:|---:|---|
| 131,072 | 40 | 131,072 | 5.24 M | 96 k+: 1.40 M; 64 k+: 2.68 M |
| 65,536 | 60 | 65,536 | 3.93 M | 64 k tail: 90 k; 32 k+: 2.01 M |
| 24,576 | 80 | 24,576 | 1.97 M | 16 k+: 0.69 M |
| **total** | **180** | — | **11.14 M** | ≥96 k ≈ 12.6 % of corpus; ≥32 k ≈ 54 % |

Resulting position-band mass: 0-4 k 0.72 M; 4-16 k 2.16 M; 16-32 k 2.29 M; 32-64 k 3.20 M; 64-96 k 1.24 M;
96-131 k 1.40 M. (No dedicated short streams: 180 streams already give ≥80× redundancy below 24 k; short
streams cost a boot each and buy only content diversity, which the conversation mix supplies inside long
streams.)

**Content mix per stream (all generator-side, seeded):**
1. **Mimic-lane rendered conversations (~50 % of tokens).** Lift the task builders of
   `campaign/mimicbench2.py` (40 tasks, depth axes D1-D8: doctrine absorption, voice transfer,
   architectural continuation, multi-turn correction, style switching, UI induction, internalization,
   document structure) into an OFFLINE generator: emit message lists, render through the pack's
   `chat_template.jinja`, tokenize with the pack tokenizer (`/home/svg/LLM/Custom-data/packs/iq3_s/tokenizer`,
   HF fast format: tokenizer.json + vocab/merges; serve passes specials with `parse_special=True`,
   serve/server.py:283-284). Mimicbench2 itself is HTTP-coupled — reuse its task definitions, not its runner.
   Target ≈ 1,400 conversations × ~4 k tok mean.
2. **Repo-prose study turns (~35 %).** The `depth-lane-real.py` SOURCES pattern (AGENTS.md, README,
   docs/*.md, GOAL, SVG-CUSTOM*.md — 45 files, calibrated **3.27 chars/token**, depth-lane-real.py:31-46).
   Repo prose is only ~0.7 M tokens total, so the 131 k rungs (5.24 M tokens) reuse it ~6-8× — defeated by
   seeded chunk-shuffles and different conversation wrappers (the deficit is positional; F-236 content share
   is handled by the mimic majority + real prose; LongSpec offset aug stays a P3 lever per T17F/exec-C).
3. **Seeded synthetic ladders (~15 %).** The depth-lane.py deterministic-filler pattern for structural
   diversity (lists, code blocks, tables); capped at 15 % because filler is the -14.9 pt class.

Conversations are concatenated with their terminator specials; the driver records every conversation-start
record index per shard in `seams-<shard>.json` (exec-C §2.1 contract) — the trainer drops cross-seam pairs
(F-77 chunk-seam law). Generation order, shuffles and any sampling carry per-stream seeds
(`seed = blake2s(shard_id)`, numpy PCG64), recorded in `manifest.json` → fully reproducible token files.
The driver reports the fraction of stream tokens OUTSIDE the 40,525-row draft sub-vocab (draft_vocab.bin);
exec-C's STOP line is >2 %.

**Boots: 180** (40+60+80). At 40-90 s load+init each (probe measures; page cache keeps the 84 GB arena warm
between boots when the lane is down) that is 2-4.5 h of overhead over a ~35-45 min compute core — one
overnight window. Driver = `campaign/harvest_build.py` (corpus) + `campaign/harvest_run.py` (loop: df-gate,
boot, validate shard, next) + `campaign/harvest_check.py` (§7 checks). Naming: harvest boot files live in
`/home/svg/LLM/Custom-data/harvest/` (exec-C's pinned input dir).

**Boot flags: production parity minus serve-only machinery (decided):** mirror
`custom-iq3_s-8081.json` + the launcher's exported STRATA_* env (GDN-chunked et al. change prefill numerics —
the corpus must be production-class states), MINUS `--serve`/conversation-cache/batch slots, MINUS `--mtp`
(the dump is target-only; `on_chunk` stays free for the dump lambda; no drafter VRAM/time), MINUS
`--expert-profile*` (adaptive tier is decode-side), PLUS `--dump-final-r` + `STRATA_DUMP_FINAL_R_BATCH=1` +
`--max-new 1 --greedy` + `--tokens-file`. KEEP: `--pack`, `--native`, `--ple-gguf`, `--expert-cache auto`,
`--max-context 131072`, `--kv int8 --kv-resident 32768`, `--prefill auto:32768`, `--spec 6`, `--spec-min-p 0.5`,
`--pcie-frac 0.35 --pcie-mode dma`, `--vram-reserve-mib 2000`, `--control-vector-scaled refusal-r.gguf:1.0
--control-vector-layer-range 4 24 --cvec-mode project --cvec-dir per-layer`, `--ple-io mmap`. The cvec and KV
flags are NOT optional: production R includes the refusal cvec (layers 4-24) and int8-KV attention — the
drafter must train on the states it will actually see (the F-104/F-105 class).

---

## 5. THE THROUGHPUT PROBE RUNBOOK (30-40 min, before ANY corpus build commits disk)

```bash
cd /home/svg/LLM/strata-custom-2
# 0) build the patched engine (§2), confirm the marker string:
cmake --build build --target strata -j16 && strings engine/strata | grep -c "batched final-R dump armed"

# 1) probe streams (driver builds from the §4 mix; deterministic, seeded):
python3 campaign/harvest_build.py --probe --lens 32768,131072 \
    --out-dir /home/svg/LLM/Custom-data/harvest/tokens/

# 2) lane down first (harvest owns the GPU), then per probe length L in {32768, 131072}:
set -a; source <(grep '^export ' launch-custom-iq3s-8081.sh); set +a
/usr/bin/time -v env STRATA_DUMP_FINAL_R_BATCH=1 ./engine/strata \
  --pack /home/svg/LLM/Custom-data/packs/iq3_s \
  --native /home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf /home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --expert-cache auto --max-context 131072 --kv int8 --kv-resident 32768 \
  --prefill auto:32768 --spec 6 --spec-min-p 0.5 --max-new 1 --greedy \
  --pcie-frac 0.35 --pcie-mode dma --vram-reserve-mib 2000 \
  --control-vector-scaled refusal-r.gguf:1.0 --control-vector-layer-range 4 24 \
  --cvec-mode project --cvec-dir per-layer --ple-io mmap \
  --tokens-file /home/svg/LLM/Custom-data/harvest/tokens/probe-$L.txt \
  --dump-final-r /home/svg/LLM/Custom-data/harvest/records-probe-$L.f32 \
  2> /home/svg/LLM/Custom-data/harvest/logs/probe-$L.log

# 3) READ (from the log):
#    boot/load seconds  (first timestamp delta)          -> boot-count budget
#    "strata generate: prompt chunk auto: N tokens"      -> the chunk the auto scan took
#    "strata generate: prefill P tokens in C chunks, X ms (Y tok/s)"   (generate.cpp:13877)  -> THE rate
#    "Maximum resident set size" (time -v)               -> RAM table row
python3 campaign/harvest_check.py records-probe-$L.f32 tokens/probe-$L.txt   # §7 checks, exits nonzero on any miss
```

**What to read:** the prefill line's `tok/s` at 32 k and at 131 k (deep chunks pay KV streaming + long-context
attention; production measured 6291-6972 t/s at 32-128 k on this rig, and the serve lane did 122,877 tokens
in 16,997 ms = 7229 t/s — campaign/results/depth-t17f-real120k.txt). Expect the dump tax to land the probe
at ~5.5-6.5 k t/s (32 k) and ~4-5.5 k t/s (131 k).

**Decision rule:**
- **GO** if rate(32 k) ≥ 4,500 t/s AND rate(131 k) ≥ 2,500 t/s. Budget: 11.14 M / blended ≈ 4.5-5.5 k
  ⇒ 34-41 min compute.
- **FIX-THEN-GO** if rate(32 k) < 4,500: profile the delta (pinned staging via `cudaHostAlloc` for `rows`,
  then async copy on a side stream, then per-chunk `fflush` elision). One fix attempt, re-probe.
- **FALLBACK (S25, zero-code)** if the patched lane cannot beat ~2,000 t/s blended or the patch is blocked:
  run the SAME boots with `STRATA_PREFILL_DUMP_R_ALL=/path.bf16` (prefill.cpp:4193) and reconstruct
  `(pos, tok)` driver-side (rows are in position order; per stream: row count must equal n_prompt-1 — the
  last prompt position is simply absent, drop nothing else). A second 10-line reader mode in the trainer.
- **NO-GO arm does not exist for "per-token fallback":** for our pack it writes zero records (§1) — do not
  schedule the T14B "22 h per-token" arm; that number assumed a live token loop.

---

## 6. DISK / RAM / TIME, FILE LAYOUT, CLEANUP

| Item | Size | Basis |
|---|---:|---|
| records f32, 11.14 M pos × 40,976 B | **451 GB** | §1 layout; `/home` has 705 GB free (measured 10-18; T14B's 735 is stale) |
| (bf16 arm, only if disk demands) | 226 GB | {i64,i64,bf16[10240]} = 20,496 B; S25 conversion cost applies |
| probe shards + logs | < 2 GB | 32 k + 131 k |
| tokens/*.txt corpus (kept) | ~120 MB | ~11 M ids as decimal text |
| torch env + head/embd train-cache + export chain (exec-C) | ~12 GB | exec-C §2.5/§4 |
| **peak disk** | **~465 GB of 705** | 240 GB margin; df-gate per shard at 90 % |

RAM during a boot: mmapped pack arena ~84 GB (page cache; warm between boots with the lane down), pinned KV
~1.5 GB, dump staging ≤ 1.34 GB (one 32 k-chunk of f32), engine working set ~5 GB → **< 100 GB of 184 GB**.
The production lane must be DOWN (VRAM + the RAM guard); harvest never coexists with 8081.

Wall clock: 180 boots × (load 40-90 s + compute 25-72 s + validate ~2 s) ≈ **3-5 h total**, resumable per
shard (append-only, validated then manifest-sealed). bf16 arm would add ~10-25 % wall (host conversion).

**File layout (trainer read pattern = random W-record windows at real pos ids → fixed record size ⇒ pread):**
```
/home/svg/LLM/Custom-data/harvest/
  manifest.json               # engine build hash + git HEAD, dtype, record bytes (40976),
                              # per-shard: {file, n_records, n_tokens, lens, seed, content mix, off-subset %}
  seams-<shard>.json          # sorted int64 record indices where a conversation starts (F-77 drops)
  records-<shard>.f32         # 180 shards, one per stream (shard = stream keeps fds/preads bounded)
  tokens/<shard>.txt          # pretokenized stream (the alignment oracle + corpus reproducibility)
  windows.npy                 # exec-C §2.1's one-pass index (built by the trainer, not harvest)
  logs/<shard>.log            # per-boot engine log (the prefill line is the rate ledger)
  results -> campaign/results/harvest-<label>.txt   # one headline file per campaign law
```

**Cleanup:** after exec-C's P4 value-audit AND the G0.2 stair pass: delete `records-*.f32` (rebuildable from
tokens/ + the patched engine + the recorded seeds — that is the archive). Keep manifest, seams, tokens/,
windows.npy, logs, probe logs forever (KB-MB scale).

---

## 7. RISKS AND GATES (each with its one-line check)

1. **Record-contract drift (THE silent one).** Exec-C §2.1 pins `{i32,i32}`/40,968 B from T14B §2.2's stale
   read; the engine writes `{i64,i64}`/40,976 B (generate.cpp:13971). A trainer built to 40,968 misaligns
   every record after the first — finite floats, plausible values, garbage training. *Check:* build the
   reader to 40,976 and assert `tok[i] == tokens_file[i]` on 1,000 random indices per shard
   (`harvest_check.py`); exec-C takes a dated revision note.
2. **Row misalignment / short writes.** Any skipped or duplicated row shifts the stream. *Check:* per shard:
   `file_size % 40,976 == 0`; record count == token count; `pos` strictly +1 from 0; the §1 invariants.
3. **Conversation-cache skip.** Verified structural: the checkpoint/park machinery is serve-only
   (generate.cpp:1572-1583, :8374-8519) and the serve loop returns at :13753 BEFORE the dump path exists;
   CLI generate has no cache — every boot re-reads its stream from position 0 (the production contrast is
   visible in depth-t17f-real120k.txt: "123992 reused + 19 read in 76 ms" — that path dumps nothing and
   cannot occur here). *Check:* harvest is CLI-only, `--serve` never in the harvest command, logs show the
   whole stream read ("0 reused" is implicit in CLI); if a serve-lane harvest is ever attempted, unique
   prompts per request are MANDATORY (a cached prefix skips prefill and dumps nothing for the reused span).
4. **Endianness.** Records are raw little-endian host memory; trainer runs on this same x86-64 box.
   *Check:* record 0 has `pos == 0, tok == tokens[0]`; no byte-swapping code anywhere.
5. **The dead per-token lane (native pack).** Unpatched runs yield empty files; also never boot harvest with
   a layer split (on_chunk then runs on another thread/device, prefill.hpp:159-160). *Check:* the §2 marker
   line in every shard log; harvest command has no `--layer-split` and uses `--gpu 0` (single GPU).
6. **S25-fallback alignment.** Headerless bf16 rows: driver must assert rows == n_prompt-1 per stream and
   remember the missing last position. *Check:* row count equality; documented off-by-one per stream.
7. **State-parity drift (F-104 class).** Missing cvec / wrong kv / missing launcher env = distribution shift
   the G0 tap cannot see. *Check:* harvest flags diffed against custom-iq3_s-8081.json in the driver
   (whitelist); one 1 k-stream dumped with and without cvec must differ (cvec active proof).
8. **Disk exhaustion mid-run.** 451 GB steady write. *Check:* df-gate before each shard at 90 % of budget;
   shard-at-a-time resumability.
9. **Boot-count blowout.** If load+init is ~3 min (not 40-90 s), 180 boots ≈ 9 h. *Check:* probe's load
   seconds; if > 150 s, collapse the ladder to 28 × 131,072 + 44 × 65,536 (≈ 9 M positions, deep share held)
   and re-run the disk math (367 GB).
10. **fwrite truncation on interruption.** Partial last shard. *Check:* shard sealed only after
    `harvest_check.py` passes; unsealed shards are re-run (append-mode reopen only after truncate-restart).

---

## 8. Findings by confidence

**High (code/file read at HEAD this session, or measured in-repo)**
- Record = `{int64 pos, int64 tok, f32[4×2560]}` = 40,976 B; per-token hook at generate.cpp:13969-13973;
  fopen :13768; option :638/:2158. (Correction to T14B §2.2 and exec-C §2.1.)
- The hook is CLI-generate-only (serve returns :13753) and DEAD for native packs (loop break :13892; verify
  windows only, :6067 comment; window summaries in production logs).
- `Prefill::on_chunk(R_rows, T, p0)` delivers the chunk's final rows device-side after a stream sync
  (prefill.cpp:4215-4226; prefill.hpp:138-141); the MTP draft_kv lambda at generate.cpp:13825-13831 is the
  chaining precedent; window-side `final_R_all()` D2H precedent at :14431-14441.
- `--tokens-file` pretokenized input (usage :780-783); pack tokenizer dir with tokenizer.json +
  chat_template.jinja (`/home/svg/LLM/Custom-data/packs/iq3_s/tokenizer`).
- Production boot recipe (custom-iq3_s-8081.json; launcher env incl. STRATA_GDN_CHUNKED et al.).
- F-235/F-236 depth split: 53.1 % deep (fresh, real) vs 93.6 % (KV-reuse continuation) = -40.5 pt
  position-primary, -14.9 pt content-secondary (SVG-CUSTOM.md:8089-8093, F-236 commit 315919bc,
  campaign/results/depth-t17f-real120k.txt).
- 7229 t/s serve prefill at 122,877 tokens (same file); 6291-6972 t/s campaign measurements (T14B table).
- S25 hook `STRATA_PREFILL_DUMP_R_ALL` works today (prefill.cpp:4193-4214), bf16 RNE, headerless, no last
  position; `STRATA_PREFILL_DUMP_R` (every 64th, f32+pos only) at :4180-4192.
- Disk 705 GB free; rig 184 GB RAM (df/free); engine binary `engine/strata`, target `strata`
  (build/build.ninja:2248).

**Medium (multiple independent in-repo sources)**
- Corpus sizing ~11 M records sufficing for v1 (R3's 8-16 M, RH's 41 M-token precedent, exec-C's S-runs at
  16 M, gates catch under-training); deep-oversample ladder shares (derived here, tunable knobs).
- Boot overhead 40-90 s (multiple prior boot logs exist but were not re-timed this session) — the probe
  measures it before commitment.

**Low (inference, flagged)**
- Dump tax 5-8 % of prefill wall (arithmetic from memcpy+fwrite bytes vs measured chunk time; not measured).
- "~1,400 conversations ≈ 5.5 M tokens" from the 4 k mean (mimicbench2 task lengths vary).

**Unknown (searched, not found)**
- Any prior consumer of `STRATA_PREFILL_DUMP_R_ALL` (no S25 entry in SVG-CUSTOM.md, no campaign script
  references it — grep across campaign/, docs/, SVG-CUSTOM.md; the hook predates the fork import,
  git log -S shows no adopting commit in this repo). Its provenance comment ("S25, draft-layer distillation
  data") is the only documentation.
- A serve-mode dump path (none exists; final_r is unreferenced in the serve block).

---

## 9. Contradictions and resolutions

- **Exec-C §2.1 "40,968 B / int32" vs the engine's actual 40,976 B / int64.** Resolved by reading the fwrite
  (:13971). Exec-C must take a dated revision (its §2.1 explicitly says "pinned for leg A" — this file is
  leg A's counter-signature with the corrected pin). All exec-C downstream math that depends on record bytes
  (disk, page-cache, IO rates) shifts by 8/40,968 ≈ 0.02 % — immaterial except the alignment itself.
- **T14B "per-token hook, 22 h fallback" vs the native-pack dead zone.** Resolved: the per-token arm cannot
  run for our pack at all; the true fallback is S25 (§5). T14B's 45-min batched estimate STANDS via on_chunk.
- **"Depth deficit is positional" (F-235 closeout, F-236) vs "decay is CONTENT" (the crashed session's first
  read).** Resolved by F-236's corrected decomposition: position primary (-40.5 pt), content secondary
  (-14.9 pt) — both real. The corpus plan acts on both: deep-position ladder + content-matched majority;
  exec-C independently carries the training-side lever (w_deep).
- **The 93.6 % "shallow" arm being at ~124 k positions** (it is a KV-reuse continuation, not a shallow
  position): noted rather than resolved — either reading (position vs reuse-state warmth) makes FRESH deep
  reads the deficit class, which is exactly what the ladder oversamples and what the harvest records (every
  stream is a fresh read from 0).
- **T14B's "735 GB free" vs 705 today.** Resolved by measurement; corpus sized to 451 GB with a 240 GB margin
  and the bf16 arm named.

---

## 10. Open questions

1. Actual load+init seconds per CLI boot (probe measures; drives the §7.9 ladder collapse).
2. Whether the auto chunk scan takes 32,768 at 131 k context under the harvest boot (log line decides; the
   record math is chunk-independent).
3. mimic-conversation token-yield per task type after jinja rendering (driver reports; mix percentages are
   knobs, invariants are the deep-band shares).
4. Whether the off-subset fraction of mimic/ladder text vs the 40,525-row draft vocab stays < 2 % (driver
   reports; exec-C's STOP line).
5. (v2, exec-C) rejection-replay telemetry riding these insertion points.

## 11. Implications

- P1 is now a bounded overnight: ~35-line patch, 180 seeded boots, 451 GB, one validation script; no core-lib
  changes, no serve risk, the standing lane untouched (it must be down for VRAM/RAM, which is the scheduling
  constraint, not a code one).
- Exec-C can code its reader against 40,976 B TODAY and validate against the probe shards before the full
  run — the corpus contract is closed end to end.
- The corpus bakes F-235/F-236 in by construction: deep-band share ~12.6 % (≥96 k) and ~54 % (≥32 k) against
  a natural distribution near zero; if the G0.2 stair still shows a deep deficit after training, the next
  lever is the ladder shape (more 131 k streams), not new infrastructure.
- The S25 hook's existence means the harvest is never blocked on the patch: worst case the corpus ships in
  bf16 headerless rows with driver-side reconstruction, at a documented one-position-per-stream loss.

---

## 12. Sources

In-tree, read at HEAD this session: `src/program/generate.cpp` (:636-638 option doc; :13753 serve return;
:13768-13769 fopen/buffer; :13771 n_batched; :13825-13831 MTP on_chunk; :13839 prefill.run; :13872-13875
pos_start/tok/ple_prev; :13877-13882 prefill stats line; :13890-13892 native_pack break; :13969-13973 the
per-token record write; :14157 mtp.bind final_R_all; :14314-14315 p/x; :14358 window loop; :14365 first
window T=1; :14403 window[0]=x; :14421 ver.run; :14431-14441 window hashes D2H; :7332/:8374-8519/:1572-1583
serve cache; :2926 native_pack); `src/prefill/prefill.cpp` (:1406 HCN; :4180-4214 STRATA_PREFILL_DUMP_R /
_DUMP_R_ALL; :4215-4226 on_chunk call); `include/strata/prefill/prefill.hpp` (:49-56 draft_kv; :129 run;
:138-141 on_chunk contract; :159-160 stage threading); `include/strata/core/layer.hpp` (:453-456 R layout);
`custom-iq3_s-8081.json`; `launch-custom-iq3s-8081.sh` (env arms); `campaign/results/depth-t17f-real120k.txt`;
`campaign/mimicbench2.py`; `campaign/depth-lane-real.py`; `SVG-CUSTOM.md` F-235 entries (:8074-8122);
`SVG-CUSTOM-TODO.md` (W2 item 4 :1066-1071; STEP 0 results :1238+; retrain-v2 :1255-1257);
`campaign/research/T14B-fastmtp-training-infra.md`; `campaign/research/T14B-exec-C-trainer-export-spec.md`
(§2.1 contract, §2.4 w_deep, §3.3 replay); `build/build.ninja` (:2248 target strata); `df -h`, `free` via
T14B + re-measured df; `/home/svg/LLM/Custom-data/mtp/mtp-inventory.md` (+ .json); pack tokenizer dir
listing; `git log` (F-235/F-236 commits, -S search for the S25 hook).

## 13. Methods note (what did NOT work / dead ends)

- T14B's cited dump lines (13683-13688, 10320-10332, 14146/14320-14330) are all shifted ~85-110 lines at
  current HEAD — re-grep before quoting line numbers into any new spec; this file's numbers are HEAD-pinned.
- `git log -S STRATA_PREFILL_DUMP_R_ALL` shows no adopting commit in this fork (the hook arrived with the
  import snapshot); searching SVG-CUSTOM.md for "S25" returns nothing — the hook is undocumented outside its
  own comment. Do not spend more time on its provenance.
- No serve-mode dump exists; any plan that replays the corpus against the 8081 lane is dead on arrival (both
  because of the cache skip and because final_r is never opened under --serve).
- The per-token fallback arm (T14B's 22 h number) is not schedulable for this pack — removed from the plan
  rather than re-estimated.
