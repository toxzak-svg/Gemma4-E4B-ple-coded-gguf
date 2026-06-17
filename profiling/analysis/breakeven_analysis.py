#!/usr/bin/env python3
"""
Breakeven Analysis for PLE-Coded GGUF (JAX-compatible)
Addresses Open Question 2: At what hollowing ratio does PLE adapter compute cost
outweigh backbone matmul savings?

Key insight: PLE adapters are small matmuls (rank R << hidden_dim H).
Backbone matmul savings come from aggressive quantization + pruning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_matmul_cost(seq_len: int, hidden_dim: int, batch_size: int, bits: int) -> float:
    flops = 2 * batch_size * seq_len * hidden_dim * hidden_dim
    bits_multiplier = 1.0 if bits >= 8 else 0.5
    return flops * bits_multiplier


def compute_adapter_cost(
    seq_len: int,
    ple_dim: int,
    rank: int,
    hidden_dim: int,
    batch_size: int,
) -> float:
    down_flops = 2 * batch_size * seq_len * ple_dim * rank
    up_flops = 2 * batch_size * seq_len * rank * hidden_dim
    return down_flops + up_flops


def compute_breakeven_ratio(
    hidden_dim: int = 2048,
    ple_dim: int = 256,
    rank: int = 16,
    q4_bits: int = 4,
    q2_bits: int = 2,
    seq_len: int = 512,
    batch_size: int = 1,
) -> dict:
    baseline_cost = compute_matmul_cost(seq_len, hidden_dim, batch_size, bits=16)

    q4_cost = compute_matmul_cost(seq_len, hidden_dim, batch_size, bits=q4_bits)
    q2_cost = compute_matmul_cost(seq_len, hidden_dim, batch_size, bits=q2_bits)

    adapter_cost = compute_adapter_cost(seq_len, ple_dim, rank, hidden_dim, batch_size)

    results = {}
    for backbone_ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ple_subsidized_fraction = 0.5

        hollowed_cost = (
            ple_subsidized_fraction * backbone_ratio * q2_cost +
            (1 - ple_subsidized_fraction) * backbone_ratio * q4_cost
        )

        backbone_savings = baseline_cost - hollowed_cost
        net_savings = backbone_savings - adapter_cost

        results[f"{int(backbone_ratio*100)}%"] = {
            "baseline_cost_flops": baseline_cost,
            "hollowed_cost_flops": hollowed_cost,
            "backbone_savings_flops": backbone_savings,
            "adapter_cost_flops": adapter_cost,
            "net_savings_flops": net_savings,
            "adapter_overhead_ratio": adapter_cost / backbone_savings if backbone_savings > 0 else float('inf'),
            "profitable": net_savings > 0,
        }

    return results


def main():
    print("=" * 70)
    print("Breakeven Analysis: PLE Adapter Cost vs Backbone Matmul Savings")
    print("=" * 70)
    print()
    print("Config:")
    print("  Hidden dim: 2048 (Gemma E4B)")
    print("  PLE dim: 256")
    print("  Adapter rank: 16")
    print("  Sequence length: 512")
    print()

    results = compute_breakeven_ratio()

    print(f"{'Ratio':<10} {'Baseline':<15} {'Hollowed':<15} {'Savings':<15} {'Adapter':<15} {'Net':<15} {'OK?':<5}")
    print("-" * 90)

    for ratio, r in results.items():
        status = "YES" if r["profitable"] else "NO"
        print(f"{ratio:<10} {r['baseline_cost_flops']:<15.0f} {r['hollowed_cost_flops']:<15.0f} "
              f"{r['backbone_savings_flops']:<15.0f} {r['adapter_cost_flops']:<15.0f} "
              f"{r['net_savings_flops']:<15.0f} {status:<5}")

    breakeven = None
    prev_profitable = None
    for ratio, r in results.items():
        if prev_profitable is not None and prev_profitable != r["profitable"]:
            breakeven = ratio
        prev_profitable = r["profitable"]

    print()
    print("=" * 70)
    if breakeven:
        print(f"BREAKEVEN at backbone ratio ~{breakeven}")
        print("Above this ratio, adapter cost is dominated by backbone savings.")
    else:
        print("Adapter cost is ALWAYS less than backbone savings (good!)")
    print("=" * 70)


if __name__ == "__main__":
    main()
