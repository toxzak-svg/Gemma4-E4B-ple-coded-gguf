# PLE-Coded GGUF

**Exploiting Gemma E4B's Per-Layer Embeddings as a Compression Side-Channel**

PLE-Coded GGUF is a compression method specific to Gemma E4B that rebalances which part of the architecture carries information — hollowing the main backbone where PLE is strong and using PLE as a high-fidelity correction stream.

This is not "quantize better." It's a different architectural insight: Gemma E4B has a built-in side-channel (PLE) that already does compression-adjacent work, giving "effective 4B" capacity out of a smaller active core. PLE-Coded GGUF explicitly exploits that pathway.

## The Core Idea

Gemma E4B uses Per-Layer Embeddings (PLE) — small per-token per-layer vectors that modulate the main residual stream. The main backbone and PLE are doing genuinely different things:

- **Main stream**: carries content
- **PLE**: modulates per-layer, provides conditioning signal

Standard quantization treats the whole model as a single unit. PLE-Coded GGUF separates these two information pathways and rebalances them:

1. **Hollow** the main backbone in PLE-dominant regions (aggressive quantization + pruning)
2. **Expand** PLE's load with small per-layer adapters
3. **Encode** both in GGUF as a two-plane structure

The result: lower memory footprint, maintained or improved quality where PLE can compensate.

## Why This Only Works for Gemma E4B

Other models (Llama, Mistral, Qwen) don't have a separate per-layer embedding stream riding alongside the main residual. Their modulation happens through the main pathway itself — you can't separate and rebalance what isn't architecturally distinct.

Gemma E4B is one of the only production models where PLE is a first-class built-in pathway, not an add-on. This method requires that architectural feature to exist.

## Status

All phases implemented. Open questions resolved (Q2, Q3, Q5) with empirical/theoretical evidence.

**Key findings from mock profiling (Q3):**
- Early layers (0-8): PLE-dominant (0.5-0.8 dominance score)
- Late layers (29-34): Backbone-dominant (0.06-0.25 dominance score)  
- 11/35 layers (31%) qualify as PLE-dominant at threshold 0.5

**Key findings from breakeven analysis (Q2):**
- Adapter cost always less than backbone savings at all compression ratios
- Net FLOPs savings: ~2-3B per forward pass even at 30% backbone size

**GGUF integration path (Q5):** Python wrapper → minimal llama.cpp hooks → upstream PR

Hardware validation still pending (requires GPU with 24GB+ VRAM).

## Colab GPU Tunnel

For remote GPU validation, use the Colab runbook in `colab/README.md`. It installs the Colab dependencies, reads `HF_TOKEN` and `NGROK_AUTHTOKEN` from runtime environment variables, and launches a Gradio control panel through either ngrok or Gradio share links:

```bash
python colab/run_colab_tunnel.py --tunnel ngrok --port 7860
```

## License

See [LICENSE](LICENSE) for terms. Commercial use requires written agreement.
