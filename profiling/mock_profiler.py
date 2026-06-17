#!/usr/bin/env python3
"""
Mock Profiler — JAX implementation for synthetic PLE dominance scores.
Uses realistic synthetic data when GPU/CPU memory is insufficient for real model profiling.
"""
import sys
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import compute_channel_attribution


def generate_synthetic_ple_scores(num_layers: int = 35) -> dict[int, dict]:
    np.random.seed(42)

    results = {}
    for layer_idx in range(num_layers):
        if layer_idx < 10:
            base = 0.5 + 0.3 * np.exp(-layer_idx / 3)
            noise = np.random.normal(0, 0.05)
        elif layer_idx < 23:
            t = (layer_idx - 10) / 13
            base = 0.5 - 0.2 * t + 0.1 * np.sin(t * np.pi)
            noise = np.random.normal(0, 0.07)
        else:
            t = (layer_idx - 23) / 11
            base = 0.3 - 0.2 * t
            noise = np.random.normal(0, 0.05)

        ple_dominance = float(np.clip(base + noise, 0.05, 0.95))

        results[layer_idx] = {
            "layer_idx": layer_idx,
            "ple_dominance": ple_dominance,
            "ple_variance": float(np.random.uniform(0.1, 2.0)),
            "output_variance": float(np.random.uniform(1.0, 5.0)),
            "is_ple_dominant": ple_dominance >= 0.5,
        }

    return results


def run_mock_profiling():
    print("=" * 60)
    print("PLE-Coded GGUF — Mock Profiling (JAX/Synthetic Data)")
    print("=" * 60)
    print()
    print("Note: Running with synthetic data due to insufficient memory/CPU.")
    print("      On a machine with GPU and 24GB+ RAM, run quick_profile.py instead.")
    print()

    num_layers = 35
    print(f"[1/3] Generating synthetic PLE scores for {num_layers} layers...")

    layer_results = generate_synthetic_ple_scores(num_layers)

    ple_dominant_layers = sorted([
        l for l, v in layer_results.items() if v["is_ple_dominant"]
    ])

    results = {
        "layer_results": layer_results,
        "ple_dominant_layers": ple_dominant_layers,
        "total_layers": num_layers,
        "batches_processed": 8,
        "source": "synthetic_mock",
    }

    print(f"      PLE-dominant layers: {ple_dominant_layers}")

    print()
    print("[2/3] Computing channel attribution statistics...")
    ple_attr, backbone_attr = compute_channel_attribution(
        jax.random.normal(jax.random.PRNGKey(0), (8, 512, 2048)),
        jax.random.normal(jax.random.PRNGKey(1), (8, 512, 2048)),
    )
    print(f"      Channel attribution shape: {ple_attr.shape}")
    print(f"      Mean PLE attribution: {float(jnp.mean(ple_attr)):.4f}")

    print()
    print("[3/3] Saving results...")
    output_path = Path("profiling/outputs/mock_ple_dominance_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"      Saved to {output_path}")

    print()
    print("=" * 60)
    print("Mock Profiling Complete")
    print("=" * 60)
    print(f"PLE-dominant layers: {ple_dominant_layers}")
    print(f"Total layers: {num_layers}")
    print()
    print("Per-layer PLE dominance:")
    for ln in sorted(layer_results.keys()):
        r = layer_results[ln]
        bar = "#" * int(r["ple_dominance"] * 20)
        status = "PLE-DOM" if r["is_ple_dominant"] else "backbone"
        print(f"  Layer {ln:2d}: {r['ple_dominance']:.4f} |{bar:<20}| {status}")

    return results


if __name__ == "__main__":
    run_mock_profiling()
