# PLE-Coded GGUF — Next Steps Plan

## Current State

All 5 implementation phases complete. Open questions Q2, Q3, Q5 resolved. Q1, Q4 still require hardware validation.

**What exists:**
- Phase 1–5 implementations (profiling, hollowing, adapters, GGUF encoding, evaluation)
- Synthetic/mock profiling showing layer distribution hypothesis
- Breakeven analysis proving adapter cost feasibility
- GGUF integration path documented

**What's missing:**
- Real hardware validation (requires GPU with 24GB+ VRAM)
- End-to-end encoding pipeline connecting all phases
- Integration with actual llama.cpp
- Real-world quality benchmarks

---

## Recommended Next Steps

### 1. Hardware Validation (Critical Path)

**Why:** Q1 (PLE bandwidth capacity) and Q4 (fine-tuning stability) cannot be resolved without real hardware.

**Steps:**
1. Acquire or access GPU with 24GB+ VRAM
2. Run `profiling/quick_profile.py` on actual Gemma E4B model
3. Validate that early layers (0-8) are actually PLE-dominant
4. Run `profiling/ple_adapters/adapter.py` fine-tuning to verify stability

**Verification criteria:**
- Per-layer PLE dominance scores match mock distribution hypothesis
- Adapter fine-tuning converges without divergence
- Loss curve shows expected behavior

---

### 2. End-to-End Encoding Pipeline

**Why:** Currently phases are standalone. Need to connect profiling → hollowing → adapters → GGUF.

**Steps:**
1. Create `profiling/pipeline.py` orchestrating all phases
2. Add input validation between phases
3. Add progress reporting and checkpointing
4. Create single command: `python -m profiling.pipeline --model google/gemma-4-E2B-it`

**Mermaid diagram:**
```mermaid
graph LR
    A[Gemma E4B] --> B[Phase 1: Profiling]
    B --> C[PLE Dominance Scores]
    C --> D[Phase 2: Hollowing]
    D --> E[Hollowed Weights + Masks]
    E --> F[Phase 3: Adapters]
    F --> G[Fine-tuned Adapters]
    G --> H[Phase 4: GGUF]
    H --> I[Two-Plane GGUF File]
    I --> J[Phase 5: Evaluation]
    J --> K[TemporalBench Results]
```

---

### 3. llama.cpp Integration (Phase 4b)

**Why:** The theoretical GGUF extension path needs implementation.

**Steps:**
1. Implement Python wrapper layer first (lowest risk)
2. Add minimal llama.cpp hooks if wrapper overhead is unacceptable
3. Submit upstream PR if llama.cpp maintainers are receptive

**Implementation:**
- Create `profiling/gguf_encoder/wrapper.py`
- Implement GGUF reading + PLE adapter application
- Benchmark vs FP16 baseline

---

### 4. Real-World Quality Benchmarks

**Why:** Need to prove PLE-Coded quality on actual tasks.

**Steps:**
1. Select benchmark tasks: TemporalBench, causal reasoning, staleness detection
2. Compare: PLE-Coded vs Q4_K_M baseline vs FP16 teacher
3. Measure: perplexity, task accuracy, memory usage, latency

**Success criteria:**
- PLE-Coded quality within 2% of FP16 on TemporalBench
- Memory reduction ≥ 30% vs Q4_K_M baseline
- Adapter overhead < 10ms latency per token

---

### 5. Documentation & Polish

**Why:** Project is implementation-complete but not user-ready.

**Steps:**
1. Add README.md with quick-start guide
2. Document command-line interfaces for each phase
3. Add requirements.txt with pinned dependencies
4. Create example scripts for common workflows

---

## Priority Order

```
1. Hardware Validation          [BLOCKING — required for Q1, Q4]
2. End-to-End Pipeline          [Enables full workflow]
3. llama.cpp Wrapper           [Validates Q5 approach]
4. Quality Benchmarks           [Proves effectiveness]
5. Documentation               [Polish for usability]
```

---

## Open Questions Status

| Q# | Question | Status | Next Action |
|----|----------|--------|-------------|
| Q1 | PLE bandwidth capacity | **PENDING** | Hardware run with real model |
| Q2 | Adapter breakeven | **RESOLVED** | No further action needed |
| Q3 | Layer distribution | **RESOLVED** (mock) | Validate on real hardware |
| Q4 | Fine-tuning stability | **PENDING** | Hardware run with fine-tuning |
| Q5 | GGUF integration path | **RESOLVED** | Implement wrapper |

---

## Resource Requirements

| Step | Time | Hardware | Dependencies |
|------|------|-----------|---------------|
| Hardware Validation | 2-4 hours | 24GB+ GPU | HuggingFace access |
| End-to-End Pipeline | 1-2 days | CPU only | All phase code |
| llama.cpp Wrapper | 3-5 days | CPU only | llama.cpp knowledge |
| Quality Benchmarks | 1-2 days | GPU | Benchmark datasets |
| Documentation | 1 day | None | Writing |
