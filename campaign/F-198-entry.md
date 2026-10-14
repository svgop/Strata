### F-198 — THE SCOREBOARD EXTENDED: pristine upstream v0.1.44 and hokar3361/Strata measured — upstream stock is strong, hokar ≡ upstream on CUDA (their patch is Metal-only), current leads stock same-window on M1/pair (2026-10-15 ~01:20, goal round 17 execution)

**THE ASK (operator).** Add two engines to the scoreboard: "strata upstream engine original" and
https://github.com/hokar3361/Strata.

**THE ENGINES.**
- **upstream-0144** = pristine upstream tag v0.1.44 from a worktree of OUR repo (the exact base the
  current binary was merged onto — the cleanest "what did the custom fork add" row).  Built
  `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`; zero source fixes needed (0.1.44
  compiles clean on GCC).  `~/LLM/strata-upstream-0144`.
- **hokar** = `git clone` of hokar3361/Strata = **exactly v0.1.44 + 3 commits** (2 README + ONE
  patch, `metal/src/*` only: 116 lines wrapping a >maxBufferLength model as overlapping no-copy
  Metal buffers — the macOS 41.75 GB cap fix that keeps 47 GB IQ3_XXS resident on 64 GB Macs).
  On CUDA the metal sources are not in the build: the two binaries are the same size
  (75,064,848 B) and same code; hashes differ only by embedded build paths.  `~/LLM/strata-hokar`.
- Both: stock boot, ZERO env battery, **v1's config recipe verbatim** (same GGUFs/pack/mtp/profile,
  spec 4 / min-p 0.7 / pcie 0.35 / kv-resident 32768 / cvec / max-context 262144 — all flags are
  upstream features; boot accepted everything).  Artifacts shared via symlinks into v1's tree.

**THE BATTERY** (second window of the night, 3+3+3, no collapses; current re-run in-window for
drift normalization):

| boot (same hour) | M1 | pair | 6-conc |
|---|---|---|---|
| upstream stock v0.1.44 | 147.1 | 149.1 | 187.7 |
| hokar3361 (v0.1.44 + metal) | 155.3 | 158.3 | 194.8 |
| current (fixed, current recipe) | **164.8** | **160.5** | 192.7 |

Run detail matters: upstream's boot WARMED through its runs (M1 141.1/144.9/155.2 — cold expert
cache on runs 1-2), hokar's cooled slightly (159.0/154.9/151.9); the distributions overlap.
**VERDICT: hokar ≡ upstream on this box** — the ~5% boot-level gap is the documented boot
asymmetry (F-188 class), not their patch: the only diff does not compile into the CUDA build.
Their reported 29-52→58-59 t/s gain is real WHERE DESIGNED (macOS buffer cap) and irrelevant here.

**READINGS.**
1. **Upstream stock 0.1.44 is a strong engine**: 147/149/188 with zero custom work — well above
   the 0.1.43-era custom v1 (117.5/108.2/129.0 in window 1).  The custom fork's early arms were
   Windows-era costs; upstream has since absorbed the easy classes (COMMIT_BATCH etc.).
2. **Current vs stock, same window, each at its own best recipe**: M1 +12.0%, pair +7.6%,
   6-conc +2.7%.  The concurrent edge is compressed tonight (trough window; at adoption windows
   the batch-6/F-31 means ran 216-class vs this window's 188-195 band) — the M1/pair edge holds
   everywhere.  vs **v1** (window 1): +37/+46/+54% stands.
3. M3b/accept accounting lines did not print on the stock servers (the probe reads those from a
   fork-side stats surface; diagnostic-only absence, noted).
4. Config-class note: reference rows run max-context 262144 (v1's recipe) vs the current
   engine's 131072 — F-182 measured that difference at −0.2..−1.3% (noise-class).

**THE SCOREBOARD (same night, two adjacent windows; current appears in both as the normalizer):**

| engine | M1 | pair | 6-conc | window |
|---|---|---|---|---|
| v1 custom (0.1.43, F-64, Windows era) | 117.5 | 108.2 | 129.0 | 1 |
| v2-frozen custom (0.1.43.1, F-172) | 132.9 | 138.1 | 186.6 | 1 |
| current @ current recipe | 161.4 | 157.7 | 198.7 | 1 |
| upstream stock v0.1.44 | 147.1 | 149.1 | 187.7 | 2 |
| hokar3361 (≡ upstream on CUDA) | 155.3 | 158.3 | 194.8 | 2 |
| current @ current recipe | **164.8** | **160.5** | **192.7** | 2 |

Reference engines kept: `~/LLM/strata-custom-v1`, `~/LLM/strata-frozen-baseline`,
`~/LLM/strata-upstream-0144` (worktree), `~/LLM/strata-hokar` (clone) — the four-generation set.
Sources: campaign/results/f198-{up,hk,cc}-{1,2,3}.txt.
