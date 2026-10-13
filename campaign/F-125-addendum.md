
**6. ADDENDUM (17:36) — a contention-poisoned reading, recorded as such.** The health probe taken right
after the lane restore read 37.8 t/s on the code class (which had measured 184 t/s twenty minutes
earlier). It is NOT a degradation capture: the engine log of that boot shows the parallel session
running its own battery on the same lane at the same time - dozens of interleaved requests (prompts of
53-312 tokens, 1-token health probes, prompt-cache hits of 11,001 tok/s) - and my request shared the
process with them. Per the Proof Law this reading is void, and it is recorded as void rather than
counted. It is the third contention-poisoned measurement today (16:33 battery, 17:15 watch, 17:33
health), which is itself the strongest remaining evidence for the external-contention explanation of
F-119: under a shared lane the engine's own `pool + plan` term ranged **5.7 - 16.1 ms/window** across
one boot (vs 1.2-1.4 ms/window on the quiet 17:17 boot), while `wait for the GPU` stayed at 9.5-10.7 ms.
The GPU side is stable; the host/pool side is what moves when the machine is shared.
