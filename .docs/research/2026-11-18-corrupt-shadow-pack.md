---
type: research
owner: .docs/research/2026-11-18-corrupt-shadow-pack.md
verdict: PROVEN — canonical Q2_0 pack corrupt on disk since the estate move; the NaN #879 class fully attributed; engine exonerated
date: 2026-11-18
---

# The corrupt shadow pack — root cause of the NaN #879 class

## Finding

`/mnt/DATA/LLM/Models/GGUF/strata/data/packs/q2_0/experts.bin` (33,973,862,400 B) carries
**non-finite fp16 scales inside its Q2_0 blocks**. Real weights never do. Structural spot-check
(48 random 16 KiB regions, block size 18 B = 2 B fp16 scale + 16 B 2-bit quants, seed 11):

- 96 / 43,680 sampled scales non-finite (NaN/Inf), spread across **36 of 48** regions
- corruption is SPARSE (≈0.22% of sampled blocks), not a contiguous region — the signature of
  copy/transfer-layer corruption, not a bad write at one offset
- no manifest hashes exist (format has no sha256 fields); source GGUFs
  (`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf`) are deleted; the old-home copy
  (`/home/svg/LLM/data/packs/q2_0`) was moved, not copied — **no clean copy of this quant survives**

## Causality — the boot timeline (logs/custom-iq3_s.log, per-boot segmentation)

| boot | store source | armed | non-finite events |
|---|---|---|---|
| Nov 9–10 ×6 | /dev/shm tmpfs (pre-move bytes) | ✓ | 0 |
| Nov 10 13:37 (the bank) | old-home canonical (pre-move) | ✓ | 0 |
| Nov 18 13:48, 16:11 | none (disarmed) | ✗ | 0 |
| Nov 18 16:17 / 16:42 / 16:50 | tmpfs re-copied from ntfs3 (post-reboot 15:24 wipe) | ✓ | 29 / 19 / 9 |
| Nov 18 18:55 (the canonical re-arm attempt) | ntfs3 direct mmap | ✓ | 19 |
| every disarmed boot since | — | ✗ | 0 |

The store was clean on the old home Nov 10 (bank boot: armed, sane, 219.6/210.4/290.3 era).
The estate move (Nov 16) copied it to ntfs3 with mtimes preserved (files still read Oct 13).
Every armed boot reading a post-move source NaN'd; every pre-move or disarmed boot was clean.
**The estate copy step corrupted the file.** The corruption is on disk — not in delivery:
the bisect-era boots read it from RAM (tmpfs re-copy) and NaN'd identically.

## Exonerated by this result

- The engine binary (bank-era sha, and now behaviorally: identical binary, good bytes → sane,
  corrupt bytes → #879, disarmed → sane — the guard itself worked as designed, refusing to emit
  the corrupt windows' tokens)
- The arm path / hugetlb fall-through / `shadow_experts.cpp` delivery
- ntfs3 page-cache mmap delivery (corrupt bytes read via buffered reads too)
- The pin regime, the estate config, mtp, the cvec gate

## Consequences

1. The rollback (385937a4d) was the correct protective action; with the pack corrupt, NO arm is
   quality-preserving. The canonical-primary ladder stays correct — the artifact under it is bad.
2. The ~20% recovery (cpu-exp 4.4 → 2.0) is blocked until a clean store exists. Options:
   a. rebuild the store from the surviving source — the IQ3_S GSQ GGUF (`gsq-iq3s/`) via the
      repo pack tools (tools/strata_pack.py, iq_pack.py); carries double-quantization error on
      the ~3% of picks that miss VRAM; MMLU gate decides
   b. re-download / regenerate the GSQ-RCO Q2_0 shards if the source is public or the operator's
      pipeline can rebuild from a base model
   c. no store (standing rollback state)
3. **Systemic risk:** the same estate copy step moved every model artifact to /mnt/DATA. Sparse
   corruption in a 34 GB file demands an integrity pass over the estate (sha vs known hashes
   where they exist — mm3 pack is 224/224 sha-verified post-import; the lane's IQ3_S GGUF reads
   sane under load). The corruption mechanism (ntfs3 driver under large writes vs a one-off)
   is unidentified.

## Verification owed

- Full-file non-finite scan of experts.bin (quantifies extent; the spot check proves presence)
- Whatever recovery path is chosen: rebuild → structural scan → armed boot → France/MMLU gate
  → standing battery (the cpu-exp ≈2.0-2.5 readback then confirms the quant law on this box)
