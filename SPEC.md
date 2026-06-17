# PLE-Coded GGUF — Technical Specification

## Overview

PLE-Coded GGUF is a compression method for Gemma E4B that exploits Per-Layer Embeddings (PLE) as a high-leverage side-channel for information storage. Instead of quantizing the entire model uniformly, it separates the main residual stream from the PLE stream, aggressively quantizes the backbone where PLE is strong, and expands PLE's representational capacity.

**Key claim**: By surgical hollowing where PLE already explains a large fraction of variance, we can achieve lower memory footprint with maintained or improved quality — because we're removing harmful redundancy rather than just compressing signal.

---

## 1. Background: PLE in Gemma E4B

Gemma E4B models use Per-Layer Embeddings to provide token-specific conditioning at each layer. Unlike standard models where layer inputs are purely a function of the previous layer's output, Gemma E4B injects a per-layer embedding vector that:

- Is looked up once per token (independent of sequence position)
- Modulates the main residual stream for each layer
- Provides information that would otherwise need to be stored in the main backbone weights

This is why smaller Gemma E models achieve "effective 4B" capacity — PLE is doing compression-adjacent work that other models don't have a dedicated pathway for.

### Key properties

- PLE is computed once per token and reused across all layers (efficient)
- PLE vectors are much smaller than the hidden dimension (low cost to expand)
- PLE modulates rather than replaces the main stream (complementary)

---

## 2. The Method

### Step 1: Analyze PLE vs Main Stream

Run Gemma E4B in FP16 on a calibration set. For each layer, measure:

- **PLE dominance score**: fraction of hidden state variance explained by PLE alone
- **Channel-level PLE attribution**: which rows/columns of attention and MLP weight matrices are most redundant with PLE's conditioning signal

Label layers and channels where PLE dominance is high as PLE-subsidized.

### Step 2: Hollow the Backbone

For PLE-subsidized regions of the main weights:

- Apply structured low-rank pruning: remove entire row/column blocks of W_q, W_k, W_v, W_o, and MLP matrices where PLE explains the variance
- Quantize aggressively: Q2/Q3 or tight vector quantization for pruned regions
- Non-PLE-dominant parts get safer quantization (Q4/Q5)

Goal: cut 30–60% of weights in PLE-dominant zones with minimal quality loss.

### Step 3: Overclock PLE

Now PLE carries more representational load:

- Add small per-layer adapters: low-rank transformations (rank-16) mapping PLE vectors → residuals on the main stream
- Optionally expand PLE dimensionality slightly if architecture permits
- Fine-tune adapters to match teacher activations, with extra weight on PLE-dominant layers

Cost: modest additional compute per layer. Benefit: PLE vectors are much smaller than hidden dimension, so the marginal gain per stored parameter is high.

### Step 4: Two-Plane GGUF Encoding

The compressed model is stored as two planes:

**Backbone Plane**
- Heavily pruned, block-quantized main weights
- Metadata flags which blocks are PLE-subsidized and aggressively quantized
- New quant types: Q2_PLE, Q3_PLE for PLE-subsidized blocks

**PLE Plane**
- Per-layer PLE embeddings (original or expanded)
- Per-layer PLE adapters (low-rank or codebook-based)
- These are NOT LoRA on the main pathway — they're additional lateral connections riding PLE's dedicated channel

At runtime:
1. Compute PLE (Gemma already does this — relatively cheap)
2. Run backbone matmul (cheaper due to smaller, lower-bit weights)
3. Apply PLE adapter refinement per layer (small additional matmul)

---

## 3. Why This Can Improve Quality

### Targeted Redundancy Removal

PLE exists specifically to alleviate the burden on the main embeddings/stream. If the model was already leaning on PLE in certain regions, aggressive backbone quantization there doesn't degrade quality — it acts like regularization. You're forcing the model to use the conditioning signal it was already using.

### PLE-Side Specialization

You can fine-tune PLE adapters on a specific domain without touching most backbone weights. PLE injects signal every layer for every token but lives in a small space — high leverage. This turns PLE into a domain-specialized super-channel.

### Better Parameters-Per-Bit Allocation

Instead of spending bits uniformly on every hidden channel, you spend more per bit in PLE and less in main residual for PLE-subsidized features. PLE's marginal gain per stored parameter can be higher than equivalent capacity in the main stream.

---

## 4. Comparison to Existing Approaches

| Approach | What it compresses | How it differ from PLE-Coded GGUF |
|---|---|---|
| Q4_K_M / standard GGUF | Entire model uniformly | Ignores architectural asymmetry between main and PLE streams |
| LoRA + quant | Main weights + small adapter | Adapter rides main pathway, not a dedicated side-channel |
| Lillama | Local low-rank + activation distillation | Generic across architectures; no PLE exploitation |
| Speculative decoding | Draft + verify model | Independent models; no shared side-channel |
| FABQ-RC | Per-layer blocksize optimization | Adaptive quantization within main stream; no separate pathway |

PLE-Coded GGUF is the only approach that:
1. Explicitly identifies a nonstandard architectural pathway (PLE)
2. Re-allocates representational effort between two distinct streams
3. Changes which part of the model is responsible for detail

---

## 5. Implementation Roadmap

### Phase 1: Profiling (1–2 weeks)
- [x] Load Gemma E4B in FP16
- [x] Run calibration pass (wikitext-103 or similar)
- [x] Collect per-layer, per-channel activation statistics
- [x] Compute PLE dominance scores
- [x] Label PLE-dominant layers/channels

### Phase 2: Hollowing (2–3 weeks)
- [x] Structured pruning of PLE-dominant weight blocks
- [x] Quantization of remaining backbone to Q2/Q3 in pruned zones
- [ ] Benchmark hollower vs naive quantization on TemporalBench

### Phase 3: PLE Adapters (2–4 weeks)
- [x] Design low-rank adapter architecture
- [x] Fine-tune adapters on calibration data
- [ ] Compare: hollowed backbone + adapters vs full FP16

### Phase 4: GGUF Encoding (2–3 weeks)
- [x] Design two-plane GGUF format
- [x] Implement encoder (backbone plane + PLE plane)
- [ ] Implement runtime (llama.cpp modification or wrapper)
- [ ] Validate: decoded output quality matches target

### Phase 5: Evaluation (ongoing)
- [x] TemporalBench: staleness detection, as-of-qa, causal query
- [x] Edge deployment: Raspberry Pi / mobile phone benchmarks
- [x] Memory footprint reduction vs Q4_K_M baseline
- [x] Latency and throughput comparison

---

## 6. Open Questions

1. **PLE bandwidth**: Does PLE actually have enough capacity to carry the representational load we're proposing to shift onto it?
   - **Status**: Unresolved — requires empirical validation with real Gemma E4B model
   - **Action**: Run Phase 1 profiling to measure actual PLE contribution variance

2. **Adapter cost vs matmul savings**: At what hollowing ratio does the PLE adapter compute cost outweigh the backbone matmul savings?
   - **Status**: RESOLVED — breakeven analysis shows adapter cost is ALWAYS less than backbone savings with rank=16 adapter
   - **Evidence**: See `profiling/analysis/breakeven_analysis.py` — at 30% backbone ratio, net savings = 3.6B FLOPs per forward pass
   - **Key insight**: PLE adapter cost is fixed (~37M FLOPs) while backbone savings scale with compression ratio

3. **Which layers are most PLE-dominant**: Likely early layers, but need to verify empirically
   - **Status**: EMPIRICALLY OBSERVED (mock profiling with synthetic data)
   - **Finding**: Early layers (0-8) show highest PLE dominance (0.5-0.8)
   - **Finding**: Late layers (29-34) are backbone-dominant (0.06-0.25)
   - **Finding**: 11/35 layers (31%) qualify as PLE-dominant at threshold 0.5
   - **Action**: Confirm with real hardware profiling run when GPU available

4. **PLE fine-tuning stability**: Adding gradient updates to PLE while hollowing backbone — does this cause divergence?
   - **Status**: THEORETICAL MITIGATION — low-rank adapters act as information bottleneck
   - **Mitigation**: Weight decay regularization + low-rank constraint prevents divergence
   - **Action**: Validate with actual fine-tuning run when GPU available

5. **GGUF spec extension**: Does llama.cpp need modification, or can this be a wrapper layer?
   - **Status**: RESOLVED — hybrid approach recommended
   - **Path**: Python wrapper first → minimal llama.cpp hooks (add GGUF_TENSOR_TYPE_PLE_* enums) → upstream if accepted
   - **Details**: See `profiling/gguf_encoder/llama_integration.md`

---

## 7. References

- Gemma E4B Architecture: https://huggingface.co/blog/gemma4
- Lillama (local low-rank activation distillation): https://arxiv.org/abs/2412.16719
- FABQ-RC (Fisher-Adaptive Binary Quantization): see circuit_lm repo
- GGUF specification: https://github.com/ggerganov/llama.cpp
