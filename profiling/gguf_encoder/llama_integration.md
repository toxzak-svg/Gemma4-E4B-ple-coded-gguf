# GGUF/llama.cpp Integration Path

## Open Question 5: Does llama.cpp need modification, or can this be a wrapper layer?

**Answer: Hybrid approach — minimal llama.cpp modification + wrapper layer**

## Analysis

### Option A: Wrapper Layer Only
- ✅ No llama.cpp modification required
- ✅ Can work with existing llama.cpp binary
- ❌ Must maintain parallel PLE plane management
- ❌ Decoding must happen in two separate code paths
- ❌ Harder to optimize PLE adapter matmuls

### Option B: Full llama.cpp Modification
- ✅ Native support for two-plane structure
- ✅ Optimal kernel fusion for PLE adapters
- ❌ Significant engineering effort
- ❌ Must maintain patch against upstream llama.cpp
- ❌ Longer time to production

### Option C: Minimal llama.cpp Hooks (Recommended)
- ✅ Small modification: add PLE tensor slots to model loading
- ✅ Existing quantization paths reused
- ✅ PLE adapter matmuls as post-process kernels
- ✅ Can upstream to llama.cpp if accepted
- ⚠️ Requires coordination with llama.cpp maintainers

## Recommended Implementation Path

### Phase 4a: Wrapper Layer (Weeks 1-2)
```
1. Create GGUF loader that reads two-plane format
2. Load backbone plane → standard llama.cpp decode
3. Apply PLE adapters as post-processing
4. Implement in Python for rapid iteration
```

### Phase 4b: llama.cpp Integration (Weeks 3-4)
```
1. Add GGUF_TENSOR_TYPE_PLE_EMBEDDING enum
2. Add ple_tensor struct to gguf_header
3. Implement ple_adapter_matmul kernel (simple fused matmul)
4. Submit PR to llama.cpp (upstream potential)
```

## Implementation Notes

### GGUF Format Extension
```c
// New tensor type (add to gguf.h)
enum gguf_tensor_type {
    // ... existing types ...
    GGUF_TENSOR_TYPE_PLE_EMBEDDING = 100,
    GGUF_TENSOR_TYPE_PLE_ADAPTER = 101,
};

// New metadata keys
#define GGUF_KEY_PLE_DIM "ple_dim"
#define GGUF_KEY_PLE_RANK "ple_rank"
#define GGUF_KEY_HAS_PLE_PLANE "has_ple_plane"
```

### Runtime Changes
```c
// In llama_model_load
if (model_header.has_ple_plane) {
    ple_embeddings = load_ple_embeddings(ctx);
    ple_adapters = load_ple_adapters(ctx);
}

// In decoder forward (post-processing)
for (layer = 0; layer < n_layers; layer++) {
    // Standard backbone decode
    compute_layer(hidden_states[layer]);
    
    // PLE adapter refinement
    if (has_ple_plane) {
        ple_vec = ple_embeddings[layer][token_idx];
        residual = ple_adapter_matmul(ple_vec, ple_adapters[layer]);
        hidden_states[layer] += residual;
    }
}
```

## Verification

To validate the decoded output quality matches target (as per SPEC.md Phase 4):
1. Compare perplexity of encoded model vs FP16 baseline
2. Run TemporalBench tasks on both
3. Verify memory savings match theoretical calculations

## Next Steps

1. Start with Python wrapper (lowest risk)
2. Benchmark wrapper performance
3. If performance acceptable → ship as-is
4. If adapter overhead significant → pursue minimal llama.cpp hooks
