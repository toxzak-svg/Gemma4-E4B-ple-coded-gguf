from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class LayerStats:
    layer_idx: int
    ple_dominance_score: float
    ple_attribution_by_channel: jnp.ndarray
    variance_explained_by_ple: float
    total_variance: float
    ple_variance: float
    backbone_variance: float
    residual_variance: float


@dataclass
class ChannelStats:
    layer_idx: int
    channel_idx: int
    ple_attribution: float
    backbone_attribution: float
    is_ple_dominant: bool


@dataclass
class ProfilingResults:
    layer_stats: list
    ple_dominant_layers: list[int]
    ple_dominant_channels: dict[int, list[int]]
    calibration_samples: int
    model_name: str
    hidden_dim: int
    num_layers: int


@dataclass
class PLEDominanceConfig:
    variance_threshold: float = 0.5
    channel_threshold: float = 0.6
    calibration_seq_len: int = 512
    calibration_batch_size: int = 4
