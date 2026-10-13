# Raw primary-source quotes — draft head designs (acceptance length vs acceptance rate)

Collected 2026-10-12. Every quote carries its URL. This is the working digest behind
`.docs/research/draft-head-designs-accept-length.md`.

## EAGLE (arXiv 2401.15077) — https://ar5iv.labs.arxiv.org/html/2401.15077

### THE METRIC DEFINITIONS (Section 4 "Metrics")
> "Like other speculative sampling-based methods, EAGLE primarily focuses on latency rather than throughput. We assess acceleration effects using the following metrics:
> - Walltime speedup ratio: The actual test speedup ratio relative to vanilla autoregressive decoding.
> - Average acceptance length τ: The average number of tokens accepted per forward pass of the target LLM.
> - Acceptance rate α: The ratio of accepted to generated tokens during drafting, gauges draft accuracy. It's less applicable for tree drafts due to multiple tokens sampled per location with only one accepted. Hence, when measuring this metric, we utilize chain drafts without tree attention, aligning with speculative sampling and DistillSpec. EAGLE's draft model inputs feature and token sequences. Autoregressive feature processing can propagate errors, so we measure the acceptance rate as n-α, considering n features predicted by the draft model, potentially with inaccuracies."

KEY: acceptance rate is explicitly called "less applicable for tree drafts". Strata runs tree/chain hybrids.

### Tree depth
> "EAGLE creates a tree-structured draft using tree attention, generating a draft tree with depth m and more than m tokens through m forward passes. For instance, as shown in Figure 6, EAGLE drafts a 10-token tree with just 3 forward passes. The actual tree structure employed by EAGLE is detailed in Appendix A.1."

> "Tables 1 and 2 indicate that in EAGLE, the target LLM generates 3.2-4.5 tokens per forward pass, surpassing vanilla decoding which produces only one token per forward pass, thereby significantly increasing generation speed."

### Accuracy comparison vs Medusa / Lookahead
> "However, their effectiveness is limited by the lower accuracy of the resulting drafts, with Medusa achieving an accuracy of about 0.6, and Lookahead even lower. In contrast, our method attains an accuracy of approximately 0.8."

### Table 2 — Average acceptance length τ and acceptance rate α on MT-bench
| T | Model | τ | 0-α | 1-α | 2-α | 3-α | 4-α |
|---|---|---|---|---|---|---|---|
| 0 | Vicuna 7B | 3.94 | 0.79 | 0.74 | 0.72 | 0.73 | 0.67 |
| 0 | Vicuna 13B | 3.98 | 0.79 | 0.74 | 0.72 | 0.74 | 0.70 |
| 0 | Vicuna 33B | 3.68 | 0.74 | 0.69 | 0.67 | 0.67 | 0.66 |
| 0 | LLaMA2-Chat 7B | 3.62 | 0.76 | 0.69 | 0.67 | 0.68 | 0.68 |
| 0 | LLaMA2-Chat 13B | 3.90 | 0.77 | 0.69 | 0.69 | 0.70 | 0.71 |
| 0 | LLaMA2-Chat 70B | 3.81 | 0.75 | 0.69 | 0.65 | 0.64 | 0.64 |
| 1 | Vicuna 7B | 3.17 | 0.71 | 0.68 | 0.66 | 0.66 | 0.65 |
| 1 | Vicuna 13B | 3.20 | 0.73 | 0.68 | 0.68 | 0.67 | 0.69 |
| 1 | Vicuna 33B | 3.22 | 0.71 | 0.67 | 0.64 | 0.64 | 0.64 |
| 1 | LLaMA2-Chat 7B | 3.30 | 0.71 | 0.66 | 0.66 | 0.66 | 0.64 |
| 1 | LLaMA2-Chat 13B | 3.45 | 0.73 | 0.69 | 0.66 | 0.67 | 0.67 |
| 1 | LLaMA2-Chat 70B | 3.46 | 0.73 | 0.67 | 0.64 | 0.66 | 0.65 |

NOTE: τ ≈ 3.9 with α in 0.67-0.79. Acceptance rate and acceptance length are NOT the same number and do not track 1:1.

### Table 1 — Speedup ratio and average acceptance length τ (HumanEval / GSM8K / Alpaca)
T=0: V7B 3.33x/4.29, 3.01x/4.00, 2.79x/3.86 · V13B 3.58x/4.39, 3.08x/3.97, 3.03x/3.95 · V33B 3.67x/4.28, 3.25x/3.94, 2.97x/3.61 · LC7B 3.17x/4.24, 2.91x/3.82, 2.78x/3.71 · LC13B 3.76x/4.52, 3.20x/4.03, 3.01x/3.83 · LC70B 3.52x/4.42, 3.03x/3.93, 2.97x/3.77
T=1: V7B 2.39x/3.43, 2.34x/3.29, 2.21x/3.30 · V13B 2.65x/3.63, 2.57x/3.60, 2.45x/3.57 · V33B 2.76x/3.62, 2.77x/3.60, 2.52x/3.32 · LC7B 2.61x/3.79, 2.40x/3.52, 2.29x/3.33 · LC13B 2.89x/3.78, 2.82x/3.67, 2.66x/3.55 · LC70B 2.92x/3.76, 2.74x/3.58, 2.65x/3.47

### Table 5 — τ with chain vs tree drafts (MT-bench, T=0)
Vicuna 7B 3.20 → 3.94 (+0.74) · Vicuna 13B 3.23 → 3.98 (+0.75) · Vicuna 33B 2.97 → 3.68 (+0.71) · LLaMA2-Chat 7B 3.00 → 3.62 (+0.62) · LC 13B 3.18 → 3.90 (+0.68) · LC 70B 3.12 → 3.81 (+0.69)
> "The implementation of tree draft and verification in EAGLE results in an approximate increase of 0.6-0.8 in the average acceptance length and about 0.3-0.5 in the speedup ratio."
> "the improvement in the speedup ratio is less pronounced than the increase in average acceptance length."

### Hardware / wall-clock (real single-GPU numbers)
> "with gpt-fast (PyTorch Labs 2023), EAGLE accelerates LLaMA2-Chat 7B decoding to 160.4 tokens/s on a single RTX 3090 GPU."
Table 4 (LLaMA2-Chat 7B, MT-bench, T=0): Vanilla (HuggingFace) 24.5 tokens/s · gpt-fast 55.1 tokens/s FP16, 106.9 tokens/s int4 · EAGLE + gpt-fast 100.2 tokens/s FP16, 160.4 tokens/s int4.
> "For the LLaMA2-Chat 70B model, EAGLE trains a decoder layer with fewer than 1B parameters ... The training is completed in 1-2 days on 4x A100 (40G) GPUs. The training of EAGLE on 7B, 13B and 33B models can even be conducted on a RTX 3090 node in 1-2 days."

### Mixtral MoE (relevant: Strata runs an MoE)
> "This modest acceleration, compared to models like LLaMA, is due to a shorter average acceptance length and the complexity of accelerating MoE models via speculative sampling. MoE models typically require reading the weights of only two experts per token during vanilla autoregressive decoding. However, during the verification phase of speculative sampling, processing multiple tokens may necessitate accessing the weights of more than two experts, contrasting with dense decoder-only models where all weights are read regardless of the number of tokens forwarded."
Table 3 (Mixtral 8x7B Instruct, MT-bench, T=0): Speedup 1.50x, τ 3.25, 0-α 0.67.

### EAGLE is lossless; Medusa/Lookahead are not
> "Reliability: EAGLE does not involve any fine-tuning of the original LLM, and the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings. This is in sharp contrast to Lookahead and Medusa which either focus solely on greedy settings or do not guarantee the preservation of distribution in these settings."

## Medusa (arXiv 2401.10774) — https://arxiv.org/html/2401.10774v3

### Typical acceptance (Section 2.3.1) — the exact criterion Strata's rescue implements
> "we propose employing a typical acceptance scheme to select plausible candidates rather than using rejection sampling."
> "Our objective is to choose candidates that are typical, meaning they are not exceedingly improbable to be produced by the original model. We use the prediction probability from the original model as a natural gauge for this and establish a threshold based on the prediction distribution to determine acceptance."

The condition:
p_original(x_{n+k} | x_1..x_{n+k-1}) > min( ε, δ·exp( −H( p_original(· | x_1..x_{n+k-1}) ) ) )
> "where H(·) denotes the entropy function, and ε, δ are the hard threshold and the entropy-dependent threshold respectively."
> "During decoding, every candidate is evaluated using this criterion, and a prefix of the candidate is accepted if it satisfies the condition. To guarantee the generation of at least one token at each step, we apply greedy decoding for the first token and unconditionally accept it while employing typical acceptance for subsequent tokens. The final prediction for the current step is determined by the longest accepted prefix among all candidates."

NOTE the structure: a MIN of a fixed floor and an entropy-scaled term. Strata's "probability floor" is ε (Medusa's "hard threshold"); the entropy term δ·exp(−H) is the second half Strata's rescue does not obviously implement.

> "when the temperature is set to 0, it reverts to greedy decoding, as only the most probable token possesses non-zero probability. As the temperature surpasses 0, the outcome of greedy decoding will consistently be accepted with appropriate ε, δ, since those tokens have the maximum probability, yielding maximal speedup. Likewise, in general scenarios, an increased temperature will correspondingly result in longer accepted sequences"

### Head count / depth choice (Section 2.2.3)
> "Empirically, we found that five heads are sufficient at most. Therefore, we recommend training with five heads and referring to the strategy described in Section 2.3.3 to determine the optimal configuration of the tree attention. With optimized tree attention, sometimes three or four heads may be enough for inference. In this case, we can ignore the redundant heads without overhead."

### Architecture
> "given the original model's last hidden states h_t at position t, we add K decoding heads to h_t. The k-th head is used to predict the token in the (t+k+1)-th position of the next tokens"
> "The cumulative number of new tokens is calculated as Σ_{k=1}^{K} Π_{i=1}^{k} s_i."

### Speedup
Abstract (https://arxiv.org/abs/2401.10774): "Medusa-1 can achieve over 2.2x speedup without compromising generation quality, while Medusa-2 further improves the speedup to 2.3-3.6x."
HTML v3 body: "Medusa can achieve a speedup of 2.3 to 2.8 times across different prompt types without compromising on the quality of generation."
(DISCREPANCY between the v3 listing abstract and the v3 HTML body — report both.)

> "Our experiments primarily focus on scenarios with a batch size of one, which is representative of the use case where LLMs are locally hosted for personal use."
> "The training only takes a few hours (e.g., 5 hours for Medusa-1 on Vicuna 7B model with a single NVIDIA A100 PCIE GPU to train on 60k ShareGPT samples)."

## Hydra (arXiv 2402.05109) — https://arxiv.org/html/2402.05109v2

### THE CLEANEST ACCEPTANCE-LENGTH DEFINITION FOUND
> "Instead, we report the average throughput, which is the number of tokens generated per second, and the average acceptance length, which is the number of tokens generated per decoding step, to evaluate the speed and quality of Hydra decoding."

### The typical acceptance criterion, restated (Section 6.3)
p_base(x̂_{t+i} | ...; τ) > min( ε, α·exp( −H( p_base(· | ...; τ) ) ) )
> "where ε is known as the posterior threshold, α is known as the posterior alpha, τ is the sampling temperature, and H(·) is the entropy. Both ε and α are hyperparameters to be tuned."

### What typical acceptance is FOR
> "The purpose of the typical acceptance verification criterion is to sample more diverse and creative sequences than greedy acceptance, while preserving the efficiency benefits of speculative decoding by avoiding the degradation in acceptance rate observed when employing rejection resampling."

### Hydra setup (the epsilon sweep — the closest published analogue of the Strata decision)
> "We fix the sampling temperature τ=0.7, vary the posterior threshold ε ∈ {0.05, 0.1, 0.15, 0.2, 0.25}, and set the posterior alpha as α = sqrt(ε)."
> "We plot how varying the posterior threshold affects both the average speculation length and the quality of the resulting generations in Figure 4."
> "For Medusa, Hydra, and Hydra++, increasing the posterior threshold slightly decreases the average s[peculation length]" (truncated)
> "We demonstrate that sampling using the typical acceptance criterion allows Hydra++ to achieve the same quality as sampling from the base model while preserving the throughput benefits of speculation."

NOTE: Hydra sweeps ε over 0.05-0.25 — far below Strata's 0.5-0.7 floor. And Hydra evaluates the sweep on TWO axes: acceptance length AND judged quality.

### Sequentially dependent heads (the Hydra idea)
> "All draft heads to date make predictions only as a function of the base model's hidden states from previously verified tokens, making them unaware of earlier tokens in the current candidate continuation. Because of the strong statistical dependencies between neighboring tokens in language, this sequential independence limits the prediction accuracy of existing draft head architectures."
p_draft(x̂_{t+i} | x_≤t, x̂_{t+1}, ..., x̂_{t+i-1}) = f_{Hydra,i}( h_{t-1}, x_t, x̂_{t+1}, ..., x̂_{t+i-1} )
> "This simple design change leads to significantly better speculation quality as compared to standard draft heads, increasing the average candidate continuation acceptance length by up to 0.46 tokens."

### Depth / tree choice (Section 4) — the throughput-driven tree search
> "We derive our decoding trees in a data-driven manner using a two-stage algorithm: first, we find a sequence of "proposal" trees T_1,...,T_N with sizes 1,...,N such that each proposal tree approximately maximizes expected acceptance length given its size; then, we choose the optimal tree size for a given setup by empirically measuring the end-to-end throughput achieved using each T_i, and selecting the tree which maximizes throughput."
> "In practice, we set the maximum tree size as N=100, and use a 100-question subset of the Alpaca dataset to gather our simulated acceptance length and throughput statistics."

KEY DESIGN PATTERN: acceptance length is used to *propose* tree shapes; THROUGHPUT is the *selection* criterion. Depth is chosen by wall clock, not by a ratio.

### Hardware and numbers (Section 5, 6.1)
> "We benchmark all 7B and 13B parameter experiments on a single A100-40GB GPU and all 33B parameter experiments on a single A100-80GB GPU."
> "Training is performed on 8x NVIDIA A100-80GB GPUs"
Vicuna 7B/13B/33B, MT-Bench, greedy acceptance:
- Hydra: 2.36x / 2.17x / 2.15x vs autoregressive; 1.11x / 1.10x / 1.11x vs Medusa
- Hydra++: 2.70x / 2.50x / 2.53x vs autoregressive; 1.27x / 1.27x / 1.31x vs Medusa
> "for batch size 1 Hydra++ has a 2.70x improvement in throughput compared to standard decoding, but this gain decreases to 1.63x at batch size 8."
> "While we find that all speculative decoding techniques outperform standard autoregressive decoding for all batch sizes examined, the relative improvement of speculative decoding decreases as the batch size increases."
> "at larger batch sizes it is easier for verification to become compute-bound, and the number of tokens verified per sequence per step must be more tightly controlled to avoid saturating the GPU's compute capacity and entering the regime where speculative decoding becomes unprofitable."

### The mechanism link between the two metrics
> "For all verification criteria in common use, using a draft model whose predictions more accurately match those of the base model will result in increased average acceptance lengths, and consequently greater decoding throughput."

(Note: this says acceptance LENGTH, not acceptance RATE, is the quantity that converts to throughput.)

## Lookahead decoding (arXiv 2402.02057 + LMSYS blog)
- https://arxiv.org/abs/2402.02057
- https://www.lmsys.org/blog/2023-11-21-lookahead-decoding/

### Headline claim — steps, not tokens
Abstract: "It allows trading per-step log(FLOPs) to reduce the number of total decoding steps, is more parallelizable on single or multiple modern accelerators ... Our implementation of Lookahead decoding can speed up autoregressive decoding by up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs in code completion tasks."

### Acceptance rate named as the CAP (the field's own framing)
> "First, the maximum speedup that speculative decoding based methods can achieve is limited by the token acceptance rate, or equivalently, how accurately the draft model can predict the main model's outputs."
> "Lookahead decoding is able to generate n-grams each step, as opposed to producing just one token, hence reducing the total number of decoding steps -- generating N tokens in less than N steps."
> "Linearly reduces the number of decoding steps relative to log(FLOPs) per step."

### Parameters and how depth is chosen (window W, n-gram N, cap G)
> "window size W: how far ahead we look in future token positions to conduct parallel decoding."
> "N-gram size N: how many steps we look back into the past Jacobi iteration trajectory to retrieve n-grams."
> "To manage the cost, we set a cap of G on the number of candidate n-grams considered in the verification branch. In practice, we often set this cap proportional to W (e.g., G=W)."
> "The FLOPs needed for each lookahead decoding step are proportional to the number of input tokens per step, which is the sum of the lookahead branch size and the verification branch size: W*(N-1) + G*(N-1)."
> "when N is large enough, an exponential increase in the W can result in a linear reduction of decoding steps. Thus, we can achieve linear compression of the steps by trading exponentially more FLOPs since we set G=W."

Table 1 — "Good configurations for window size W and N-gram size N on A100":
| Model | Window size W | N-gram size N |
|---|---|---|
| 7B | 15 | 5 |
| 13B | 10 | 5 |
| 33B | 7 | 5 |

> "Our experimental results show that on A100, the following configs in Table 1 work well in most cases. The 7B, 13B, and 33B models require 120x, 80x, and 56x extra FLOPs per step, respectively. However, because of the memory-intensive bound characteristic of the LLM decoding, these extra FLOPs only bring little per-step cost and a visible step compression ratio, resulting in a notable speedup."
> "if W and N are too large, each lookahead decoding step might be too costly and slow down the decoding despite reducing decoding steps. Increasing N together with W would be best to achieve balanced performance"

### Headline numbers (LMSYS blog)
- LLaMA-Chat on MT-Bench: "roughly 1.5x speedup across several model settings"
- CodeLLaMA on HumanEval: "more than 2x latency reduction"
- CodeLLaMA-Instruct on GSM8K: "1.8x latency reduction"
- Hardware: "The 7B, 13B, and 33B models are evaluated on a single A100 GPU, and the 70B model is evaluated on two A100 GPUs with pipeline parallelism, all under fp16 precision."
- "lookahead decoding achieves speedup without any finetuning or draft models"
- "Given this property, lookahead decoding should be used in scenarios where latency is vital, e.g., surplus FLOPs exist that can be traded for latency"

### EAGLE-3's independent measurement of Lookahead (Table 1, Vicuna 13B, T=0)
Mean over MT-bench/HumanEval/GSM8K/Alpaca/CNN-DM: Lookahead 1.62x, τ 1.67 (the lowest τ of the six methods measured).

## EAGLE-3 (arXiv 2503.01840) — https://arxiv.org/html/2503.01840v3
### The cross-system comparison table (Table 1 caption)
> "Table 1: Speedup ratios and average acceptance lengths τ of different methods. V represents Vicuna, L31 represents LLaMA-Instruct 3.1, L33 represents LLaMA-Instruct 3.3, and DSL represents DeepSeek-R1-Distill-LLaMA. SpS denotes standard speculative sampling, with its draft model being Vicuna-68M. Methods like Medusa relax acceptance conditions under non-greedy settings, which do not guarantee lossless acceleration. Therefore, we do not compare EAGLE-3 with these methods when temperature=1."

Temperature=0, Vicuna 13B, mean across 5 tasks (Speedup | τ):
SpS 1.92x | 2.24 · PLD 1.74x | 1.80 · Medusa 2.12x | 2.51 · Lookahead 1.62x | 1.67 · Hydra 2.80x | 3.50 · EAGLE 3.05x | 3.96 · EAGLE-2 4.22x | 4.83 · EAGLE-3 5.51x | 6.62

Per-task, Vicuna 13B, EAGLE-3, T=0 (Speedup | τ): MT-bench 5.58x | 6.65 · HumanEval 6.47x | 7.54 · GSM8K 5.32x | 6.29 · Alpaca 5.16x | 6.17 · CNN/DM 5.01x | 6.47

T=1, Vicuna 13B, EAGLE-3 mean: 4.65x | τ 5.67. L31 8B EAGLE-3 mean 3.45x | 4.92. L33 70B EAGLE-2 2.65x | 3.54.

NOTE: 5 of 5 systems in this table are ranked by τ and speedup together. NO system in this table carries an acceptance-rate column.

### Batch size / throughput (the wall-clock turn)
> "Speculative sampling is often thought to reduce throughput at large batch sizes. However, in SGLang, a production-grade framework, EAGLE-3 improves throughput by 40% at a batch size of 64."
Abstract: "In the SGLang framework, EAGLE-3 achieves a 1.38x throughput improvement at a batch size of 64."
> "EAGLE-3, trained with approximately 8x more data than EAGLE, achieves a 1.4x latency speedup over EAGLE-2 at batch size 1."
> "We identify that this limitation arises from EAGLE's feature prediction constraints."

### The acceptance-rate decomposition (why a single ratio hides the failure)
> "the acceptance rate 0-α of the first draft token improves significantly. However, the output of the draft model in Step 1 ... is far away from the ground-truth f_{t+1}, causing the input sequence ... in Step 2 to deviate significantly from the training distribution, resulting in a very low acceptance rate 1-α for the second draft token"

KEY: EAGLE-3 reports acceptance rate PER POSITION (0-α, 1-α, 2-α, 3-α, 4-α), not as one blended ratio. A single blended acceptance rate hides which position broke.
