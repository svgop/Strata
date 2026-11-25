# F-467 working notes (2026-11-25, from 16:59) - cache 32 GiB; the prewarm; a census correction

## The census correction (found 17:22 while reading the warm-in's burst)
The engine's per-request line (`strata serve: prompt ...`, generate.cpp:17595) is printed by the SOLO path. Under `--batch`
a request whose admission produced one token prints that line and continues in the batch windows (`BADM <slot> 1`,
generate.cpp:17473-17475): the line is an ADMISSION record ("1 generated, drafts accepted 0 of 0"), and nothing
summarises the generation that follows.  F-466's census read those lines as generations.

Production boot 11-22 15:59, the server's completion lines (`[strata] done: N tokens in X s`, server.py:6055; X = the
whole request, rate = tokens since the first token): 3,795 requests, 4,819,855 generated tokens, 80,557 request-seconds
(p50 602 tokens a request).  The engine's lines for the same boot: 1,703,151 generated tokens, 9,119 decode-seconds.
3,145 of its 4,821 lines are admission records (read p50 426 tokens on a 93,568-token context).

Corrected split of the server-recorded request-seconds: prompt reads 6,718 s = 8.3 % (cold re-reads 3,440 s = 4.3 %,
per-turn reads 2,985 s = 3.7 %); solo-lane decode 9,119 s = 11.3 %; batch-lane decode about 64,700 s = 80 %.
Per request the batch lane runs at 3,116,704 / 64,700 = 48 t/s against 187 t/s solo.  1,026 requests have no
completion line (their reads are in the 6,718 s; their time is unmeasured).

## The warm-in as it stands (ladder boot A0, server.log)
17 s pre-boot file sweep (80.1 GiB) -> engine load -> READY -> 3 s post-pin sweep -> corpus 12 s -> replay 28 s;
"up in 68 s".  Six-way burst: reads 2-5 s each (they serialise), 3-5 t/s a request at first, about 150 t/s aggregate
against 260-300 steady.  Clocks: strata-gpu-lock pins SM and memory clocks at maximum while a port listens.  THP is
madvise.  Every fresh conversation carries a ~1.1k-token system prompt, so its first read is at or above
stream_all_min (1,024) and pays the 30 GB walk: 745-850 ms.

Discard leg against the measured legs of the same boot, 20 boots in 6 batteries (all with this warm-in): M1 median
-0.0 % to -12.7 % by battery, 6-conc -0.1 % to -5.4 %.

## The cold cost by shape after the OLD warm-in (campaign/cold-cost-probe.py, boot 1 of 2, 17:38, 8 GiB cache)
Results in campaign/results/ctx-f467-cold-before/ (run.txt, serve-lines.txt).  Prompts: 800 pseudo-words = a ~3.8k-token
prompt; deep 28,571 words = ~97.5k tokens (this tokenizer cuts random letter-strings into 3.38 tokens a word).

| shape                       | first                         | again            | engine's own terms on the first                         |
|-----------------------------|-------------------------------|------------------|---------------------------------------------------------|
| solo, 256 tokens            | 168.5 t/s, TTFT 1.27 s        | 238.3 t/s        | two phases 128 + 110 tokens in 753 + 724 ms; tier hit 90.2 % then 78.3 % (2.2 %, 6.1 % of routed over PCIe) |
|                             |                               |                  | again: one phase, 256 in 1,081 ms; hit 95.5 % (1.1 %)   |
| tool turn, two tool calls   | TTFT 1.05 s, 88 tok in 621 ms | TTFT 1.22 s, 523 ms | tier hit 85.6 % (4.3 % over PCIe) against 92.8 % (2.0 %) |
| burst 2                     | 78.1 t/s aggregate, 5.12 s    | 85.5 t/s, 4.68 s |                                                         |
| burst 3 / 4 / 5 / 6         | 85.2 / 89.9 / 92.3 / 95.0     | 85.9 / 89.8 / 92.4 / 92.9 |                                              |
| deep read ~97.5k, x3        | TTFT 14.37, 14.68, 14.58 s    | -                | 6,849 / 6,745 / 6,693 tok/s: no first-use cost          |
| deep turn 1 / 2             | 726 tokens read in 761 ms; 1,481 in 1,164 ms | - |                                      |

Mechanism of the one real cost: the adaptive expert tier fits a new traffic's routing during its first request; a
miss is a RAM-to-GPU swap on the window's path (8.5 ms a window against 5.3).  The corpus warms generic prose; no
stage shows the tier a tool call.  Absolute cost: about 0.4 s on the first long generation, about 0.1 s on the first
tool call.  Nothing else the probe reaches is cold after the old warm-in.  The hole that remains is structural: a
live request arriving mid-warm-in pre-empts the rest of it (the server yields), and then the engine IS cold where the
skipped stages would have warmed it.
