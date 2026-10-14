### F-196 — THE BATCH COMMIT DOUBLE BUFFER: the exposed cudaStreamSynchronize removed from commit_slot_prefixes; adopted as hygiene below the win bar, and the boot-B pair-dip watch item CLOSED (2026-10-14 ~23:59, goal round 17 execution)

**THE ISSUE (F-194 #3 / F-195's "commit-async = small code win").** `Verifier::commit_slot_prefixes`
(verify.cpp) launched the batch commit graph and then `cudaStreamSynchronize`'d unconditionally —
the exposed host block F-194's phase table measured (~1.2 ms class, every window).  The sync existed
for one reason: the mapped `h_commitb_` is a SINGLE buffer, and the next round's `stage_batch`
rewrites it — without the sync, that write races the commit graph's `copy_i32_from_mapped` node.
The serial path solved this years ago (`set_commit_async`: launch, record `commit_done_`,
`wait_commit` later); the batch path never got it.

**THE FIX (bit-identical by construction — timing only).** Parity double buffer:
- `h_commitb_`/`m_commitb_` and the device `commitb_` arena doubled; a parity's halves at
  `p*cb_host_` / `p*cb_dev_` (verify.hpp members + init_slots).
- The parity is baked into the commit graph key (`ckey` = bkey + parity — the pointers are frozen
  at capture); `capture_commit_batch(rows, S, hbase, parity, ..)` captures BOTH parities up front
  in `stage_batch` (no mid-serving capture when the parity flips); `evict_batch_graph` frees both.
- `stage_batch` flips `commit_parity_` once per round; `commit_slot_prefixes` and `batch_launch`'s
  inline commit use the round's parity.
- The sync is replaced by: launch → record `commitb_done_` → return, under the EXISTING
  `set_commit_async` gate (so `STRATA_COMMIT_SYNC` still forces the old behavior, and a missing
  event falls back to sync).  The PREVIOUS round's event is checked at the next commit — an error
  checkpoint, not a wait.

**WHY NO SYNC IS SOUND.** A parity half is rewritten only two rounds later (stage_batch flipped
first), and every path between observes the stream drained: `run_slot_rows` syncs cs_ before the
host reads picks; `batch_poll` reports done only on `cudaStreamQuery(cs_)==cudaSuccess` — the whole
stream, commit graph included.  The drafter reads only the picks and its own K/V (the serial path's
shipped ordering, generate.cpp:9214's note); the next window graph is stream-ordered behind the
commit on cs_.  The adapt thread's swap surface (expert-cache slots) is disjoint from the commit
graph's reads (session state, staging buffers, dense ssm_norm weights).

**PROOF (Lock Law: adjacent boots C/D, anchor recipe, 3+3 same-hour probes, zero collapses).**
M1 169.0→167.3 (−1.0%), pair 141.8→145.2 (+2.4%), 6-conc 185.3→189.3 (+2.2%), M3b 79.8→80.7,
acceptance 79.2→79.4% — IDENTICAL sampling (the change touches no arithmetic; no MMLU gate due).
Boot E (fixed @ current recipe): 161.4 / 157.7 / 198.7, M3b 79.7.

**VERDICT.** Below the +4% win bar on every metric — NOT claimed as a scored win, scorecard
unchanged.  ADOPTED AS HYGIENE under the no-regression clause: nothing below −1%, both concurrent
metrics +2%+, a measured inefficiency removed, and the batch path now matches the serial path's
proven async design.  Smoke: single + concurrent pair coherent, zero errors.

**THE BOOT-B PAIR DIP (F-188 watch item) — CLOSED.** The sync's true pair cost is ~2.4% (C→D),
not the ~12% F-188's boot-B reading suggested; that 157.4 was drift asymmetry between boots.
COMMIT_BATCH stays default-on (its ~70-node reduction is real; the sync is now gone).

Rollback: `~/LLM/backup/strata-preF196-rollback` (sha 56895c58…; also /tmp copy).  Standing
launcher: dead `STRATA_EXCHANGE_ROTATE=1` arm deleted at this restart per F-195 ("exchange
rotation unavailable" every boot — behavior-neutral).

Sources: campaign/results/f197-{cu,cf,cc}-{1,2,3}.txt; smoke /tmp/smoke-f196.out.
