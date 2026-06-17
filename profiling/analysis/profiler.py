# Phase 1: Profiling — JAX/Flax implementation for PLE dominance analysis
import logging
from pathlib import Path
from typing import Optional, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_source: Literal["lmstudio", "huggingface", "local"]):
        self.model_source = model_source
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if jax.devices()[0].platform == "cuda" else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_gemma_e2b(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = "google/gemma-4-E2B-it",
    ):
        if self.model_source == "lmstudio":
            return self._load_from_lmstudio(model_path)
        elif self.model_source == "huggingface":
            return self._load_from_huggingface(model_name)
        else:
            return self._load_from_local(model_path)

    def _load_from_lmstudio(self, model_path: Optional[str] = None):
        try:
            from lmstudio import LMStudioClient
            client = LMStudioClient()
            logger.info("Connected to LM Studio")
            if model_path:
                self.model = client.model(model_path)
            else:
                models = client.list_downloaded_models()
                if models:
                    self.model = models[0]
                    logger.info(f"Loaded model: {self.model.config.identifiers}")
                else:
                    raise RuntimeError("No models available in LM Studio")
            return self.model, None
        except ImportError:
            logger.warning("lmstudio package not available, falling back to transformers")
            return self._load_from_huggingface(model_name)

    def _load_from_huggingface(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading {model_name} from HuggingFace")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
        )
        return self.model, self.tokenizer

    def _load_from_local(self, model_path: Optional[str]):
        if not model_path:
            raise ValueError("model_path required for local loading")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading from local path: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=self.device,
        )
        return self.model, self.tokenizer

    def get_model_weights(self) -> dict:
        """Extract model weights as JAX arrays."""
        if self.model is None:
            return {}
        state_dict = self.model.state_dict()
        return {k: jnp.array(v.numpy()) for k, v in state_dict.items()}


def get_calibration_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    seq_len: int = 512,
    num_samples: int = 256,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    logger.info(f"Loading calibration dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.select(range(num_samples))
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return result

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    return tokenized


class LayerActivationCollector:
    def __init__(self, model, layer_names: Optional[list[str]] = None):
        self.model = model
        self.layer_names = layer_names or []
        self.layer_inputs = {}
        self.layer_outputs = {}
        self._hook_handles = []

    def register_hooks(self, layer_names: list[str]):
        """Register hooks for specified layer names."""
        import torch
        for name, module in self.model.named_modules():
            if "model.language_model.layers." in name:
                parts = name.split(".")
                if len(parts) >= 4 and parts[-1].isdigit():
                    layer_num = int(parts[-1])
                    if f"layer_{layer_num}" in layer_names or not layer_names:
                        handle_inp = module.register_forward_hook(
                            self._create_input_hook(name), with_kwargs=True
                        )
                        handle_out = module.register_forward_hook(
                            self._create_output_hook(f"{name}_out"), with_kwargs=True
                        )
                        self._hook_handles.extend([handle_inp, handle_out])

    def _create_input_hook(self, name: str):
        def hook(module, input, output):
            self.layer_inputs[name] = input[0].detach()
        return hook

    def _create_output_hook(self, name: str):
        def hook(module, input, output):
            self.layer_outputs[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return hook

    def clear(self):
        self.layer_inputs.clear()
        self.layer_outputs.clear()

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


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
    """Compute per-channel PLE vs backbone attribution scores."""
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


def run_profiling(
    model,
    dataloader,
    device: str = "cuda",
    variance_threshold: float = 0.5,
) -> dict:
    """Alias for run_layer_profiling for backward compatibility."""
    return run_layer_profiling(model, dataloader, device, variance_threshold)


ActivationCollector = LayerActivationCollector


def run_layer_profiling(
    model,
    dataloader,
    device: str = "cuda",
    variance_threshold: float = 0.5,
) -> dict:
    """Run PLE dominance profiling on model layers."""
    import torch
    collector = LayerActivationCollector(model)
    collector.register_hooks([])
    layer_results = {}
    ple_dominant_layers = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)
            _ = model(input_ids)

            for name, layer_input in collector.layer_inputs.items():
                if "_out" in name:
                    continue
                layer_output = collector.layer_outputs.get(f"{name}_out", None)
                if layer_output is None:
                    continue

                parts = name.split(".")
                layer_num = int(parts[-1])

                if layer_num not in layer_results:
                    layer_results[layer_num] = {
                        "inputs": [],
                        "outputs": [],
                    }

                layer_results[layer_num]["inputs"].append(layer_input)
                layer_results[layer_num]["outputs"].append(layer_output)

            collector.clear()

    results = {}
    for layer_num in sorted(layer_results.keys()):
        inputs = torch.cat([x.unsqueeze(0) for x in layer_results[layer_num]["inputs"]], dim=0)
        outputs = torch.cat([x.unsqueeze(0) for x in layer_results[layer_num]["outputs"]], dim=0)

        input_mean = torch.mean(inputs, dim=(0, 1))
        output_mean = torch.mean(outputs, dim=(0, 1))

        analysis = analyze_layer_ple_dominance(
            input_mean.unsqueeze(0).numpy(),
            output_mean.unsqueeze(0).numpy(),
            layer_num,
            variance_threshold,
        )

        results[layer_num] = analysis

        if analysis["is_ple_dominant"]:
            ple_dominant_layers.append(layer_num)

    collector.remove_hooks()

    return {
        "layer_results": results,
        "ple_dominant_layers": sorted(ple_dominant_layers),
        "total_layers": len(results),
        "batches_processed": len(dataloader),
    }


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
