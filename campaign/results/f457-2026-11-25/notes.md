# F-457 working notes (one dated line per finished step)

- 2026-11-25 09:53 step 0 - read SVG-CUSTOM.md 14012-14055 and 14197-14215 only; results dir empty at start; HEAD d6d380ae3;
  operator files modified in the tree (config/custom-dsv41-8081.json, serve/chat_template.jinja, serve/server.py,
  tools/persona_vector.py, tools/refusal_subspace.py) - NOT touched.
- 09:55 step 1a DONE - trace.md (a) written: neighbours gr_up_v3_kernel (160x256, 9.38 us, same stream, gap 0.19) before; gdn_step_norm_multi_kernel (48x128, 5.70 us) next on same stream but 15 us later (projection+conv branch gates it); gdn_ab sits on a slack branch.
- 10:00 step 1b - launch site read (verify.cpp:1549-1625, side mixer=2 on sm120 = active path :1551 on side_); slack measured all instances: gdn_ab END -> step START median 16.90 us, min 5.82, binding 0/24804 (f457-slack.py)
- 10:01 step 1 DONE - trace.md complete: STOP (gdn_ab off the window's critical path: 0/24804 instances last to finish; consumer fold serializes the a/b body; producer fold needs a grid-wide barrier). Steps 2-6 not run (conditional on GO).
- 2026-11-25 10:02 step 1 CLOSED - labels re-read against the code and corrected in trace.md (gr_up_v3 = 2nd half of the stream-split
  hyper-connection READ producing xm = mixed_, verify.cpp:1485 / fused_gr.cu:2784; step launched on cs, verify.cpp:1644).
- 2026-11-25 10:03 steps 2-6 NOT RUN (each is conditional on GO): no SVG-CUSTOM.md edit (F-457 is not pre-registered there), no build,
  no deploy, no capture, no smoke, no ladder.
- CLOSEOUT (nothing was started, so nothing to restore): engine/strata sha256 3865e27a (the champion, untouched; pre-f457 backup not
  needed - no deploy happened); build-fresh/strata 7a4d3477 (baseline, untouched); engine DOWN, port 8081 free; no GPU job run.
