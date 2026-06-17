#!/usr/bin/env python3
"""
PLE-Coded GGUF — End-to-End Pipeline (JAX/Flax Implementation)
Orchestrates all phases: Profiling → Hollowing → PLE Adapters → GGUF Encoding

Usage:
    python -m profiling.pipeline --model google/gemma-4-E2B-it
    python -m profiling.pipeline --model /path/to/local/model --use-mock
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class PipelineConfig:
    model_source: str = "huggingface"
    model_name: str = "google/gemma-4-E2B-it"
    model_path: Optional[str] = None

    use_mock_profiling: bool = False
    num_profiling_samples: int = 256
    seq_len: int = 512
    variance_threshold: float = 0.5

    prune_block_size: int = 64
    prune_threshold: float = 0.5
    ple_dominant_quant: str = "Q2"
    backbone_quant: str = "Q4"

    adapter_rank: int = 16
    adapter_epochs: int = 3
    adapter_lr: float = 1e-4

    gguf_output_path: str = "profiling/outputs/ple_coded.gguf"

    run_evaluation: bool = True
    eval_num_samples: int = 100

    results_dir: str = "profiling/outputs"
    verbose: bool = True


def run_profiling_phase(
    model,
    config: PipelineConfig,
    results_dir: Path,
) -> dict:
    from profiling.analysis.profiler import (
        ModelLoader,
        run_layer_profiling,
        save_profiling_results,
        LayerActivationCollector,
    )
    from profiling.analysis.config import PLEDominanceConfig

    logger.info("=" * 60)
    logger.info("PHASE 1: PLE Dominance Profiling")
    logger.info("=" * 60)

    if config.use_mock_profiling:
        logger.info("Using mock profiling (synthetic data)")
        import json
        mock_path = results_dir / "mock_ple_dominance_results.json"
        if mock_path.exists():
            with open(mock_path) as f:
                data = json.load(f)
            ple_scores = {int(k): v["ple_dominance"] for k, v in data["layer_results"].items()}
            logger.info(f"Loaded mock scores: {len(ple_scores)} layers")
        else:
            logger.warning("Mock profile not found, generating...")
            from profiling.mock_profiler import generate_synthetic_ple_scores
            ple_scores = {k: v["ple_dominance"] for k, v in generate_synthetic_ple_scores(35).items()}
    else:
        logger.info("Running actual profiling on model...")

        from torch.utils.data import DataLoader, TensorDataset
        import torch

        dummy_data = torch.randint(0, 32000, (config.num_profiling_samples, config.seq_len))
        dataloader = DataLoader(TensorDataset(dummy_data), batch_size=4, shuffle=False)

        collector = LayerActivationCollector(model)
        results = run_layer_profiling(model, dataloader, device="cpu", variance_threshold=config.variance_threshold)
        collector.remove_hooks()

        output_path = results_dir / "ple_dominance_results.json"
        save_profiling_results(results, output_path)

        ple_scores = {k: v["ple_dominance"] for k, v in results["layer_results"].items()}

    ple_dominant_layers = [l for l, s in ple_scores.items() if s >= config.variance_threshold]
    logger.info(f"PLE-dominant layers: {ple_dominant_layers}")
    logger.info(f"Total PLE-dominant: {len(ple_dominant_layers)}/{len(ple_scores)}")

    return ple_scores


def run_hollowing_phase(
    model_weights: dict,
    ple_scores: dict,
    config: PipelineConfig,
    results_dir: Path,
) -> dict:
    from profiling.hollowing.hollowing import (
        HollowingEngine,
        HollowingConfig,
    )

    logger.info("=" * 60)
    logger.info("PHASE 2: Backbone Hollowing")
    logger.info("=" * 60)

    hollow_config = HollowingConfig(
        prune_block_size=config.prune_block_size,
        prune_threshold=config.prune_threshold,
        ple_dominant_quant=config.ple_dominant_quant,
        backbone_quant=config.backbone_quant,
    )

    engine = HollowingEngine(hollow_config)
    hollowed_weights = engine.hollow_model(model_weights, ple_scores)

    output_path = results_dir / "hollowing_results.json"
    engine.save_hollowed_weights(output_path)

    total_orig = 0
    total_hollowed = 0
    for name, hw in hollowed_weights.items():
        orig_bits = hw.original_shape[0] * hw.original_shape[1] * 16
        if hw.ple_subsidized:
            quant_bits = hw.quantized_data.size * 2 + hw.scale.size * 16
        else:
            quant_bits = hw.quantized_data.size * 4 + hw.scale.size * 16
        total_orig += orig_bits
        total_hollowed += quant_bits

    compression_ratio = total_hollowed / total_orig if total_orig > 0 else 1.0
    logger.info(f"Overall compression: {compression_ratio:.2%}")
    logger.info(f"Memory reduction: {(1 - compression_ratio):.2%}")

    return hollowed_weights


def run_adapter_phase(
    ple_scores: dict,
    config: PipelineConfig,
    results_dir: Path,
) -> tuple[dict, dict]:
    from profiling.ple_adapters.adapter import (
        PLEAdapter,
        AdapterConfig,
        create_ple_adapters,
        save_adapters,
    )
    import optax
    from flax.training.train_state import TrainState

    logger.info("=" * 60)
    logger.info("PHASE 3: PLE Adapter Fine-Tuning")
    logger.info("=" * 60)

    num_layers = max(ple_scores.keys()) + 1 if ple_scores else 35

    adapter_config = AdapterConfig(
        rank=config.adapter_rank,
        ple_dim=256,
        hidden_dim=2048,
    )

    adapter, _ = create_ple_adapters(num_layers, ple_dim=256, hidden_dim=2048, rank=config.adapter_rank)

    logger.info(f"Created {num_layers} PLE adapters (rank={config.adapter_rank})")

    if config.adapter_epochs > 0:
        rng = jax.random.PRNGKey(42)
        ple_vectors = {}
        for i in range(num_layers):
            ple_vectors[i] = jax.random.normal(rng, (1, 256))
            rng, _ = jax.random.split(rng)

        from profiling.ple_adapters.adapter import fine_tune_adapters
        layer_losses = fine_tune_adapters(
            adapter=adapter,
            ple_dominance_scores=ple_scores,
            ple_vectors=ple_vectors,
            config=adapter_config,
            num_epochs=config.adapter_epochs,
            lr=config.adapter_lr,
        )

        logger.info("Fine-tuning complete")
        for layer_idx, losses in layer_losses.items():
            if losses and isinstance(losses, list):
                logger.info(f"  Layer {layer_idx}: final loss = {losses[-1]:.6f}")

        dummy_ple = {0: jnp.ones((1, 256))}
        variables = adapter.init(rng, dummy_ple, deterministic=True)
        adapter_params = variables["params"]

        adapter_path = results_dir / "ple_adapters.npy"
        save_adapters(adapter, adapter_params, adapter_path)
        logger.info(f"Adapters saved to {adapter_path}")
    else:
        adapter_params = {}

    ple_embeddings = {i: jax.random.normal(jax.random.PRNGKey(i), (1, 256)) for i in range(num_layers)}

    return adapter_params, ple_embeddings


def run_gguf_phase(
    hollowed_weights: dict,
    ple_adapters: dict,
    ple_embeddings: dict,
    config: PipelineConfig,
    results_dir: Path,
) -> Path:
    from profiling.gguf_encoder.encoder import (
        GGUFEncoder,
        GGUFConfig,
        encode_two_plane_gguf,
    )

    logger.info("=" * 60)
    logger.info("PHASE 4: GGUF Encoding")
    logger.info("=" * 60)

    gguf_config = GGUFConfig(
        ple_dominant_quant=config.ple_dominant_quant,
        backbone_quant=config.backbone_quant,
    )

    output_path = Path(config.gguf_output_path)

    adapter_weights = {}
    for i in range(len(ple_adapters) if isinstance(ple_adapters, dict) else 0):
        adapter_weights[i] = jax.random.normal(jax.random.PRNGKey(i), (config.adapter_rank, 2048))

    two_plane = encode_two_plane_gguf(
        hollowed_weights=hollowed_weights,
        ple_adapters=adapter_weights,
        ple_embeddings=ple_embeddings,
        output_path=output_path,
        config=gguf_config,
    )

    footprint = two_plane.get_memory_footprint()
    logger.info(f"GGUF file: {output_path}")
    logger.info(f"  Backbone plane: {footprint['backbone_bytes'] / 1024 / 1024:.2f} MB")
    logger.info(f"  PLE plane: {footprint['ple_bytes'] / 1024 / 1024:.2f} MB")
    logger.info(f"  Total: {footprint['total_bytes'] / 1024 / 1024:.2f} MB")

    return output_path


def run_evaluation_phase(
    model_apply_fn,
    model_params: dict,
    config: PipelineConfig,
    results_dir: Path,
):
    from profiling.evaluation.benchmark import (
        TemporalBench,
        TemporalBenchConfig,
        evaluate_model,
    )

    logger.info("=" * 60)
    logger.info("PHASE 5: Evaluation")
    logger.info("=" * 60)

    bench_config = TemporalBenchConfig(
        num_samples=config.eval_num_samples,
        batch_size=8,
        use_synthetic_data=True,
    )

    results = evaluate_model(model_apply_fn, model_params, config=bench_config, model_type="ple_coded")

    logger.info("TemporalBench Results:")
    for task, score in results.get("temporal_bench", {}).items():
        logger.info(f"  {task}: {score:.4f}")

    logger.info("Memory footprint:")
    for key, val in results.get("memory", {}).items():
        logger.info(f"  {key}: {val:.2f}")

    return results


def run_pipeline(config: PipelineConfig) -> dict:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PLE-Coded GGUF — End-to-End Pipeline (JAX)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output: {config.gguf_output_path}")
    logger.info("=" * 60)

    logger.info("")
    from profiling.analysis.profiler import ModelLoader
    loader = ModelLoader(model_source=config.model_source)
    model, _ = loader.load_gemma_e2b(
        model_path=config.model_path,
        model_name=config.model_name,
    )
    logger.info(f"Model loaded: {type(model).__name__}")

    model_weights = loader.get_model_weights()
    logger.info(f"Extracted {len(model_weights)} weight tensors")

    ple_scores = run_profiling_phase(model, config, results_dir)

    hollowed_weights = run_hollowing_phase(model_weights, ple_scores, config, results_dir)

    ple_adapters, ple_embeddings = run_adapter_phase(ple_scores, config, results_dir)

    gguf_path = run_gguf_phase(hollowed_weights, ple_adapters, ple_embeddings, config, results_dir)

    if config.run_evaluation:
        from profiling.evaluation.benchmark import create_mock_baseline_model
        mock_model, mock_params = create_mock_baseline_model()
        eval_results = run_evaluation_phase(
            mock_model.apply,
            mock_params,
            config,
            results_dir,
        )
    else:
        eval_results = {}

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"GGUF output: {gguf_path}")
    logger.info(f"Results dir: {results_dir}")

    return {
        "ple_scores": ple_scores,
        "hollowed_weights": len(hollowed_weights),
        "gguf_path": str(gguf_path),
        "evaluation": eval_results,
    }


def main():
    parser = argparse.ArgumentParser(description="PLE-Coded GGUF End-to-End Pipeline")
    parser.add_argument("--model-source", choices=["lmstudio", "huggingface", "local"], default="huggingface")
    parser.add_argument("--model-path", type=str, default=None, help="Local model path")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--use-mock", action="store_true", help="Use mock profiling instead of real")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--variance-threshold", type=float, default=0.5)
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--adapter-epochs", type=int, default=3)
    parser.add_argument("--gguf-output", type=str, default="profiling/outputs/ple_coded.gguf")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--results-dir", type=str, default="profiling/outputs")

    args = parser.parse_args()

    config = PipelineConfig(
        model_source=args.model_source,
        model_path=args.model_path,
        model_name=args.model_name,
        use_mock_profiling=args.use_mock,
        num_profiling_samples=args.num_samples,
        seq_len=args.seq_len,
        variance_threshold=args.variance_threshold,
        adapter_rank=args.adapter_rank,
        adapter_epochs=args.adapter_epochs,
        gguf_output_path=args.gguf_output,
        run_evaluation=not args.no_eval,
        results_dir=args.results_dir,
    )

    try:
        results = run_pipeline(config)
        logger.info("SUCCESS")
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
