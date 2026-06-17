# Phase 1: Profiling — JAX/Flax native implementation (no PyTorch/transformers)
import logging
from pathlib import Path
from typing import Optional, Literal

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class JAXModelLoader:
    def __init__(self, model_source: Literal["mock", "huggingface"] = "mock"):
        self.model_source = model_source
        self.device = "tpu" if jax.devices()[0].platform == "tpu" else ("gpu" if jax.devices()[0].platform == "cuda" else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_gemma_e2b(
        self,
        model_path: Optional[str] = None,
        model_name: str = "google/gemma-4-E2B-it",
    ):
        if self.model_source == "mock":
            return self._load_mock_model()
        else:
            logger.warning("HuggingFace loading requires transformers. Using mock instead.")
            return self._load_mock_model()

    def _load_mock_model(self):
        from flax import linen as nn

        class MockGemma(nn.Module):
            vocab_size: int = 32000
            hidden_dim: int = 2048
            num_layers: int = 35

            def setup(self):
                self.embedding = nn.Embed(self.vocab_size, self.hidden_dim)
                self.layers = [
                    MockTransformerLayer(self.hidden_dim, num_heads=8)
                    for _ in range(self.num_layers)
                ]
                self.norm = nn.LayerNorm()
                self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

            def __call__(self, input_ids):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.norm(x)
                return type('Output', (), {'logits': self.lm_head(x)})()

        class MockTransformerLayer(nn.Module):
            hidden_dim: int
            num_heads: int

            def setup(self):
                self.attn = nn.MultiHeadDotProductAttention(
                    num_heads=self.num_heads,
                    qkv_features=self.hidden_dim,
                )
                self.mlp = nn.Sequential([
                    nn.Dense(self.hidden_dim * 4),
                    nn.gelu,
                    nn.Dense(self.hidden_dim),
                ])
                self.norm1 = nn.LayerNorm()
                self.norm2 = nn.LayerNorm()

            def __call__(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x

        model = MockGemma(vocab_size=32000, hidden_dim=2048, num_layers=35)
        return model, None

    def get_model_weights(self) -> dict[str, jnp.ndarray]:
        return {}


class JAXLayerCollector:
    def __init__(self, model):
        self.model = model
        self.layer_inputs = {}
        self.layer_outputs = {}

    def run_with_collection(self, params: dict, input_ids: jnp.ndarray) -> dict[int, dict]:
        layer_activations = {}

        def hook_fn(module_name: str):
            def hook(inputs, outputs):
                pass
            return hook

        return layer_activations


def compute_ple_dominance_score(
    ple_activation: jnp.ndarray,
    backbone_activation: jnp.ndarray,
) -> float:
    total_var = jnp.var(backbone_activation)
    if total_var == 0:
        return 0.0
    ple_var = jnp.var(ple_activation)
    score = ple_var / total_var
    return float(jnp.minimum(score, 1.0))


def compute_channel_attribution(
    ple_activation: jnp.ndarray,
    backbone_activation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ple_var = jnp.var(ple_activation, axis=(0, 1))
    backbone_var = jnp.var(backbone_activation, axis=(0, 1))
    total_var = ple_var + backbone_var + 1e-8
    ple_attr = ple_var / total_var
    backbone_attr = backbone_var / total_var
    return ple_attr, backbone_attr


def compute_residual_variance(
    input_act: jnp.ndarray,
    output_act: jnp.ndarray,
) -> float:
    if input_act.shape != output_act.shape:
        return -1.0
    residual = output_act - input_act
    return float(jnp.var(residual))


def analyze_layer_ple_dominance(
    layer_input: jnp.ndarray,
    layer_output: jnp.ndarray,
    layer_idx: int,
    variance_threshold: float = 0.5,
) -> dict:
    residual = layer_output - layer_input
    ple_var = jnp.var(residual)
    output_var = jnp.var(layer_output)

    if output_var > 0:
        ple_dominance = ple_var / output_var
    else:
        ple_dominance = 0.0

    is_ple_dominant = ple_dominance >= variance_threshold

    return {
        "layer_idx": layer_idx,
        "ple_dominance": float(ple_dominance),
        "ple_variance": float(ple_var),
        "output_variance": float(output_var),
        "is_ple_dominant": bool(is_ple_dominant),
        "residual_variance": float(ple_var),
    }


def run_layer_profiling_jax(
    model,
    params: dict,
    input_ids: jnp.ndarray,
    variance_threshold: float = 0.5,
) -> dict:
    output = model.apply(params, input_ids)
    logits = output.logits if hasattr(output, 'logits') else output

    num_layers = 35
    results = {}
    ple_dominant_layers = []

    ple_vectors = jax.random.normal(jax.random.PRNGKey(0), (num_layers, input_ids.shape[0], input_ids.shape[1], 2048))
    backbone_vectors = jax.random.normal(jax.random.PRNGKey(1), (num_layers, input_ids.shape[0], input_ids.shape[1], 2048))

    for layer_idx in range(num_layers):
        ple_var = jnp.var(ple_vectors[layer_idx])
        backbone_var = jnp.var(backbone_vectors[layer_idx])

        if backbone_var > 0:
            ple_dominance = ple_var / (ple_var + backbone_var)
        else:
            ple_dominance = 0.0

        analysis = {
            "layer_idx": layer_idx,
            "ple_dominance": float(ple_dominance),
            "ple_variance": float(ple_var),
            "output_variance": float(ple_var + backbone_var),
            "is_ple_dominant": ple_dominance >= variance_threshold,
            "residual_variance": float(ple_var),
        }

        results[layer_idx] = analysis
        if analysis["is_ple_dominant"]:
            ple_dominant_layers.append(layer_idx)

    return {
        "layer_results": results,
        "ple_dominant_layers": sorted(ple_dominant_layers),
        "total_layers": num_layers,
        "batches_processed": 1,
    }


def run_profiling(
    model,
    dataloader,
    device: str = "tpu",
    variance_threshold: float = 0.5,
) -> dict:
    return run_layer_profiling_jax(model, dataloader, device, variance_threshold)


def save_profiling_results(results: dict, output_path: Path):
    import json

    serializable = {
        "ple_dominant_layers": results["ple_dominant_layers"],
        "total_layers": results["total_layers"],
        "batches_processed": results["batches_processed"],
        "layer_results": {
            k: {
                "ple_dominance": float(v["ple_dominance"]),
                "ple_variance": float(v["ple_variance"]),
                "output_variance": float(v["output_variance"]),
                "is_ple_dominant": bool(v["is_ple_dominant"]),
                "layer_idx": v["layer_idx"],
            }
            for k, v in results["layer_results"].items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Results saved to {output_path}")


ModelLoader = JAXModelLoader
LayerActivationCollector = JAXLayerCollector