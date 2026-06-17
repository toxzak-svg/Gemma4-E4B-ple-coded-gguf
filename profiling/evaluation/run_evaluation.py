#!/usr/bin/env python3
"""
Evaluation CLI (JAX) — Run TemporalBench evaluation and model comparison.

Usage:
    python -m profiling.evaluation.run_evaluation --model-type ple_coded
    python -m profiling.evaluation.run_evaluation --compare
    python -m profiling.evaluation.run_evaluation --mock-model
"""
import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SimpleTransformer(nn.Module):
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 4

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.hidden_dim)
        self.layers = [
            TransformerLayer(self.hidden_dim, nhead=8)
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return type('Output', (), {'logits': self.lm_head(x)})()


class TransformerLayer(nn.Module):
    d_model: int
    nhead: int

    def setup(self):
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.d_model,
        )
        self.linear1 = nn.Dense(self.d_model * 4)
        self.linear2 = nn.Dense(self.d_model)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x):
        attn_out = self.self_attn(x, x)
        x = self.norm1(x + attn_out)
        ff = self.linear2(nn.gelu(self.linear1(x)))
        x = self.norm2(x + ff)
        return x


def create_simple_model():
    model = SimpleTransformer()
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    return model, variables["params"]


def run_ple_coded_evaluation(config):
    from profiling.evaluation.benchmark import (
        TemporalBench,
        TemporalBenchConfig,
        evaluate_model,
        save_evaluation_results,
    )

    logger.info("PLE-Coded Evaluation")
    logger.info("=" * 60)

    bench_config = TemporalBenchConfig(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        use_synthetic_data=True,
        seed=config.seed,
    )

    bench = TemporalBench(bench_config)

    model, params = create_simple_model()

    logger.info(f"Running TemporalBench with {config.num_samples} samples...")
    temporal_results = bench.run_all_benchmarks(model.apply, params, "ple_coded")

    from profiling.evaluation.benchmark import EdgeBenchmark
    edge = EdgeBenchmark()
    memory_result = edge.measure_memory_footprint(params)

    results = {
        "temporal_bench": temporal_results,
        "memory": memory_result,
    }

    logger.info("")
    logger.info("TemporalBench Results:")
    for task, score in temporal_results.items():
        logger.info(f"  {task}: {score:.4f}")

    logger.info("")
    logger.info("Memory Footprint:")
    for key, val in memory_result.items():
        logger.info(f"  {key}: {val:.2f}")

    if config.output:
        output_path = Path(config.output)
        save_evaluation_results(results, output_path)

    return results


def run_model_comparison(config):
    from profiling.evaluation.benchmark import (
        ModelComparator,
        TemporalBenchConfig,
        create_mock_baseline_model,
    )

    logger.info("Model Comparison: PLE-Coded vs Baseline")
    logger.info("=" * 60)

    bench_config = TemporalBenchConfig(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        use_synthetic_data=True,
        seed=config.seed,
    )

    ple_model, ple_params = create_simple_model()
    baseline_model, baseline_params = create_mock_baseline_model()

    comparator = ModelComparator(bench_config)
    reports = comparator.run_comparison(
        ple_model.apply, ple_params,
        baseline_model.apply, baseline_params,
        "baseline_q4"
    )
    comparator.print_comparison_report(reports)

    return reports


def run_mock_evaluation(config):
    from profiling.evaluation.benchmark import (
        TemporalBench,
        TemporalBenchConfig,
        EdgeBenchmark,
        save_evaluation_results,
    )

    logger.info("Mock Evaluation (Synthetic Data)")
    logger.info("=" * 60)

    bench_config = TemporalBenchConfig(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        use_synthetic_data=True,
        seed=config.seed,
    )

    bench = TemporalBench(bench_config)
    edge = EdgeBenchmark()

    model, params = create_simple_model()

    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Running TemporalBench with {config.num_samples} samples...")

    temporal_results = bench.run_all_benchmarks(model.apply, params, "mock_ple_coded")
    memory_result = edge.measure_memory_footprint(params)

    results = {
        "temporal_bench": temporal_results,
        "memory": memory_result,
        "model_type": "mock_ple_coded",
        "num_samples": config.num_samples,
    }

    logger.info("")
    logger.info("TemporalBench Results:")
    for task, score in temporal_results.items():
        logger.info(f"  {task}: {score:.4f}")

    logger.info("")
    logger.info("Memory Footprint:")
    for key, val in memory_result.items():
        logger.info(f"  {key}: {val:.2f}")

    if config.output:
        output_path = Path(config.output)
        save_evaluation_results(results, output_path)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PLE-Coded Evaluation CLI")

    parser.add_argument(
        "--mode",
        choices=["ple_coded", "compare", "mock"],
        default="mock",
        help="Evaluation mode",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    try:
        if args.mode == "ple_coded":
            run_ple_coded_evaluation(args)
        elif args.mode == "compare":
            run_model_comparison(args)
        else:
            run_mock_evaluation(args)

        logger.info("Evaluation complete.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
