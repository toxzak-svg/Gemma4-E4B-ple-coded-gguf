# Phase 3: PLE Adapters — JAX/Flax implementation for low-rank adapters
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    rank: int = 16
    ple_dim: int = 256
    hidden_dim: int = 2048
    dropout: float = 0.0
    ple_dominant_lr_multiplier: float = 2.0


class LowRankAdapter(nn.Module):
    config: AdapterConfig

    def setup(self):
        self.down_proj = nn.Dense(
            self.config.rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )
        self.up_proj = nn.Dense(
            self.config.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )
        if self.config.dropout > 0:
            self.dropout = nn.Dropout(rate=self.config.dropout)
        else:
            self.dropout = lambda x: x

    def __call__(self, ple_vector: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden = self.down_proj(ple_vector)
        hidden = nn.silu(hidden)
        hidden = self.dropout(hidden, deterministic=deterministic)
        residual = self.up_proj(hidden)
        return residual


class PLEAdapter(nn.Module):
    config: AdapterConfig
    num_layers: int

    def setup(self):
        self.adapters = {
            str(layer_idx): LowRankAdapter(self.config)
            for layer_idx in range(self.num_layers)
        }

    def __call__(
        self,
        ple_vectors: Dict[int, jnp.ndarray],
        deterministic: bool = True,
    ) -> Dict[int, jnp.ndarray]:
        corrections = {}
        for layer_idx, ple_vec in ple_vectors.items():
            corrections[layer_idx] = self.adapters[str(layer_idx)](ple_vec, deterministic=deterministic)
        return corrections

    def get_adapter(self, layer_idx: int) -> LowRankAdapter:
        return self.adapters[str(layer_idx)]


class AdapterTuner:
    def __init__(
        self,
        adapter: PLEAdapter,
        ple_dominance_scores: dict[int, float],
        config: AdapterConfig,
    ):
        self.adapter = adapter
        self.ple_dominance_scores = ple_dominance_scores
        self.config = config
        self.state: Optional[TrainState] = None

    def create_train_state(self, rng: jax.random.PRNGKey) -> TrainState:
        dummy_ple = jnp.ones((1, self.config.ple_dim))
        variables = self.adapter.init(rng, {0: dummy_ple}, deterministic=True)
        params = variables["params"]
        tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
        return TrainState.create(apply_fn=self.adapter.apply, params=params, tx=tx)

    def compute_adapter_loss(
        self,
        params: dict,
        ple_vector: jnp.ndarray,
        target_residual: jnp.ndarray,
        layer_idx: int,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        def apply_adapter(p):
            return self.adapter.apply({"params": p}, {layer_idx: ple_vector}, deterministic=deterministic)

        pred_residual = apply_adapter(params)
        loss = jnp.mean((pred_residual - target_residual) ** 2)
        return loss

    def fine_tune_adapters(
        self,
        ple_vectors: dict[int, jnp.ndarray],
        num_epochs: int = 3,
        lr: float = 1e-4,
        ple_dominant_weight: float = 2.0,
    ) -> dict[int, list[float]]:
        rng = jax.random.PRNGKey(42)
        self.state = self.create_train_state(rng)

        layer_losses = {}

        for epoch in range(num_epochs):
            epoch_losses = {}

            for layer_idx, ple_vec in ple_vectors.items():
                is_ple_dominant = self.ple_dominance_scores.get(layer_idx, 0.0) >= 0.5
                weight = ple_dominant_weight if is_ple_dominant else 1.0

                target = jax.random.normal(rng, ple_vec.shape)
                rng, subkey = jax.random.split(rng)

                def loss_fn(params):
                    return self.compute_adapter_loss(
                        params, ple_vec, target, layer_idx, deterministic=True
                    )

                loss, grads = jax.value_and_grad(loss_fn)(self.state.params)
                self.state = self.state.apply_gradients(grads=grads)

                if layer_idx not in epoch_losses:
                    epoch_losses[layer_idx] = []
                epoch_losses[layer_idx].append(float(loss))

            for layer_idx in epoch_losses:
                avg_loss = sum(epoch_losses[layer_idx]) / len(epoch_losses[layer_idx])
                if layer_idx not in layer_losses:
                    layer_losses[layer_idx] = []
                layer_losses[layer_idx].append(avg_loss)

            logger.info(f"Epoch {epoch+1}/{num_epochs} complete")

        return layer_losses


def create_ple_adapters(
    num_layers: int,
    ple_dim: int = 256,
    hidden_dim: int = 2048,
    rank: int = 16,
) -> tuple[PLEAdapter, AdapterConfig]:
    config = AdapterConfig(
        rank=rank,
        ple_dim=ple_dim,
        hidden_dim=hidden_dim,
    )
    adapter = PLEAdapter(config, num_layers)
    return adapter, config


def fine_tune_adapters(
    adapter: PLEAdapter,
    ple_dominance_scores: dict[int, float],
    ple_vectors: dict[int, jnp.ndarray],
    config: Optional[AdapterConfig] = None,
    num_epochs: int = 3,
    lr: float = 1e-4,
) -> dict[int, list[float]]:
    if config is None:
        config = AdapterConfig()

    tuner = AdapterTuner(adapter, ple_dominance_scores, config)
    return tuner.fine_tune_adapters(ple_vectors, num_epochs, lr)


def save_adapters(adapter: PLEAdapter, params: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        np.save(f, params)
    logger.info(f"Adapters saved to {output_path}")


def load_adapters(adapter: PLEAdapter, input_path: Path) -> tuple[PLEAdapter, dict]:
    with open(input_path, "rb") as f:
        params = np.load(f, allow_pickle=True).item()
    logger.info(f"Adapters loaded from {input_path}")
    return adapter, params
