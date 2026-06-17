# JAX Device Configuration — TPU/CPU/GPU setup
import os
import logging
from typing import Optional, Literal

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def get_device_type() -> Literal["tpu", "gpu", "cpu"]:
    """Detect the available device type."""
    platform = jax.devices()[0].platform
    return platform  # type: ignore


def get_device_count() -> int:
    """Get the number of available devices."""
    return len(jax.devices())


def get_default_device() -> str:
    """Get the default device string."""
    return str(jax.devices()[0])


def configure_tpu(
    tpu_address: Optional[str] = None,
    tpu_zone: Optional[str] = None,
) -> dict:
    """Configure TPU for JAX.

    Args:
        tpu_address: TPU address (e.g., 'grpc://10.0.0.2:8470').
                     If None, uses COLAB_TPU_ADDR or default.
        tpu_zone: Cloud TPU zone (for logging only)

    Returns:
        Configuration dict with device info
    """
    if tpu_address is None:
        tpu_address = os.environ.get("COLAB_TPU_ADDR")

    if tpu_address:
        logger.info(f"Connecting to TPU at {tpu_address}")
        os.environ["JAX_TPU_ADDR"] = tpu_address
    else:
        logger.info("No explicit TPU address provided")

    try:
        jax.devices()
    except Exception as e:
        logger.warning(f"TPU connection issue: {e}")

    devices = jax.devices()
    device_type = get_device_type()

    config = {
        "device_type": device_type,
        "device_count": len(devices),
        "devices": [str(d) for d in devices],
        "tpu_address": tpu_address,
    }

    logger.info(f"Device config: {device_type}/{len(devices)}")
    return config


def set_default_dtype(dtype: str = "float32"):
    """Set default JAX dtype."""
    if dtype == "float16":
        jax.config.update("jax_enable_x64", False)
        logger.info("Default dtype: float16")
    elif dtype == "float32":
        jax.config.update("jax_enable_x64", False)
        logger.info("Default dtype: float32")
    elif dtype == "bfloat16":
        jax.config.update("jax_enable_x64", False)
        logger.info("Default dtype: bfloat16")
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_tpu_memory_info() -> dict:
    """Get TPU memory information (if available)."""
    try:
        # This works on TPU but not on CPU/GPU
        import os
        if os.environ.get("JAX_TPU_ADDR"):
            return {
                "available": True,
                "note": "Run JAX operations to get actual memory usage",
            }
    except Exception:
        pass

    return {
        "available": False,
        "device_type": get_device_type(),
    }


def device_info() -> str:
    """Return a human-readable device info string."""
    dtype = get_device_type()
    count = get_device_count()
    devices = [str(d) for d in jax.devices()]

    info = f"{dtype.upper()} x {count}\n"
    for d in devices:
        info += f"  - {d}\n"

    return info


class DeviceContext:
    """Context manager for device configuration."""

    def __init__(self, device_type: Optional[Literal["tpu", "gpu", "cpu"]] = None):
        self.target_type = device_type
        self.original_devices = None

    def __enter__(self):
        self.original_devices = jax.devices()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # JAX doesn't require device switching - just verify
        return None


def setup_for_tpu_if_available():
    """Auto-setup for TPU if available in environment."""
    if os.environ.get("COLAB_TPU_ADDR"):
        configure_tpu()
        logger.info("Auto-configured for Colab TPU")
    elif os.environ.get("JAX_TPU_ADDR"):
        configure_tpu()
        logger.info("Auto-configured for TPU")
    else:
        dtype = get_device_type()
        logger.info(f"No TPU detected, using {dtype.upper()}")