# Phase 5: Evaluation — JAX implementation for TemporalBench benchmarks
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax

logger = logging.getLogger(__name__)


@dataclass
class TemporalBenchConfig:
    test_staleness: bool = True
    test_asof_qa: bool = True
    test_causal_query: bool = True

    num_samples: int = 1000
    batch_size: int = 8
    seq_len: int = 512

    test_raspberry_pi: bool = False
    test_mobile: bool = False

    use_synthetic_data: bool = True
    seed: int = 42


@dataclass
class EvaluationResult:
    task_name: str
    metric_name: str
    value: float
    unit: str
    model_type: str
    timestamp: float


@dataclass
class TaskSample:
    input_ids: jnp.ndarray
    expected_tokens: list[int]
    task_type: str
    metadata: dict = field(default_factory=dict)


class SyntheticTemporalDataGenerator:
    def __init__(self, vocab_size: int = 32000, seed: int = 42):
        self.vocab_size = vocab_size
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

    def generate_staleness_samples(self, num_samples: int, seq_len: int) -> list[TaskSample]:
        samples = []
        for i in range(num_samples):
            self.rng, k1, k2 = jax.random.split(self.rng, 3)
            current_token = jax.random.randint(k1, (1,), 100, 1000).item()
            stale_token = jax.random.randint(k2, (1,), 100, 1000).item()

            self.rng, k = jax.random.split(self.rng)
            input_ids = jax.random.randint(k, (seq_len,), 100, self.vocab_size)
            input_ids = input_ids.at[0].set(current_token)

            samples.append(TaskSample(
                input_ids=np.array(input_ids),
                expected_tokens=[current_token],
                task_type="staleness",
                metadata={"current": current_token, "stale": stale_token}
            ))
        return samples

    def generate_asof_qa_samples(self, num_samples: int, seq_len: int) -> list[TaskSample]:
        samples = []
        for i in range(num_samples):
            self.rng, k1 = jax.random.split(self.rng)
            input_ids = jax.random.randint(k1, (seq_len,), 100, self.vocab_size)

            self.rng, k2 = jax.random.split(self.rng)
            answer_token = jax.random.randint(k2, (1,), 100, 1000).item()

            samples.append(TaskSample(
                input_ids=np.array(input_ids),
                expected_tokens=[answer_token],
                task_type="asof_qa",
                metadata={"cutoff_aware": True}
            ))
        return samples

    def generate_causal_query_samples(self, num_samples: int, seq_len: int) -> list[TaskSample]:
        samples = []
        for i in range(num_samples):
            self.rng, k1 = jax.random.split(self.rng)
            input_ids = jax.random.randint(k1, (seq_len,), 100, self.vocab_size)

            self.rng, k2 = jax.random.split(self.rng)
            effect_token = jax.random.randint(k2, (1,), 100, 1000).item()

            samples.append(TaskSample(
                input_ids=np.array(input_ids),
                expected_tokens=[effect_token],
                task_type="causal",
                metadata={"requires_chain": True}
            ))
        return samples

    def generate_dataloader(self, config: TemporalBenchConfig) -> list[TaskSample]:
        all_samples = []
        num_per_task = config.num_samples // 3

        if config.test_staleness:
            all_samples.extend(self.generate_staleness_samples(num_per_task, config.seq_len))

        if config.test_asof_qa:
            all_samples.extend(self.generate_asof_qa_samples(num_per_task, config.seq_len))

        if config.test_causal_query:
            all_samples.extend(self.generate_causal_query_samples(num_per_task, config.seq_len))

        return all_samples


class MetricCalculator:
    @staticmethod
    def compute_token_accuracy(pred_logits: jnp.ndarray, target_ids: jnp.ndarray) -> float:
        pred_ids = jnp.argmax(pred_logits, axis=-1)

        if pred_ids.shape[-1] != target_ids.shape[-1]:
            min_len = min(pred_ids.shape[-1], target_ids.shape[-1])
            pred_ids = pred_ids[..., :min_len]
            target_ids = target_ids[..., :min_len]

        correct = jnp.sum(pred_ids == target_ids)
        total = target_ids.size

        return float(correct / total) if total > 0 else 0.0

    @staticmethod
    def compute_perplexity(logits: jnp.ndarray, target_ids: jnp.ndarray) -> float:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = target_ids.reshape(-1)
        loss = optax.softmax_cross_entropy(flat_logits, flat_targets).mean()
        return float(jnp.exp(loss))

    @staticmethod
    def compute_temporal_consistency(
        outputs_a: jnp.ndarray,
        outputs_b: jnp.ndarray,
        threshold: float = 0.05
    ) -> float:
        diff = jnp.abs(outputs_a - outputs_b).mean()
        consistency = jnp.exp(-diff / threshold)
        return float(consistency)


class TemporalBench:
    def __init__(self, config: TemporalBenchConfig):
        self.config = config
        self.results: list[EvaluationResult] = []
        self.data_generator = SyntheticTemporalDataGenerator(
            vocab_size=32000,
            seed=config.seed
        )

    def test_staleness_detection(
        self,
        model_apply_fn,
        params: dict,
        dataloader: list[TaskSample],
        model_type: str = "ple_coded",
    ) -> float:
        logger.info(f"Running staleness detection benchmark ({model_type})")

        total_correct = 0
        total_samples = 0

        staleness_samples = [s for s in dataloader if s.task_type == "staleness"]

        for sample in staleness_samples:
            input_ids = jnp.array(sample.input_ids).unsqueeze(0)

            try:
                logits = model_apply_fn(params, input_ids)

                if hasattr(logits, "logits"):
                    logits = logits.logits

                pred_token = int(jnp.argmax(logits[0, -1]))

                expected = sample.expected_tokens[0]

                if pred_token == expected:
                    total_correct += 1

                total_samples += 1

            except Exception as e:
                logger.debug(f"Sample evaluation error: {e}")
                continue

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.results.append(EvaluationResult(
            task_name="staleness_detection",
            metric_name="accuracy",
            value=accuracy,
            unit="percent",
            model_type=model_type,
            timestamp=time.time(),
        ))

        logger.info(f"  Staleness detection: {accuracy:.4f} ({total_correct}/{total_samples})")
        return accuracy

    def test_asof_qa(
        self,
        model_apply_fn,
        params: dict,
        dataloader: list[TaskSample],
        model_type: str = "ple_coded",
    ) -> float:
        logger.info(f"Running as-of-QA benchmark ({model_type})")

        total_correct = 0
        total_samples = 0

        asof_samples = [s for s in dataloader if s.task_type == "asof_qa"]

        for sample in asof_samples:
            input_ids = jnp.array(sample.input_ids).unsqueeze(0)

            try:
                logits = model_apply_fn(params, input_ids)

                if hasattr(logits, "logits"):
                    logits = logits.logits

                pred_token = int(jnp.argmax(logits[0, -1]))
                expected = sample.expected_tokens[0]

                if pred_token == expected:
                    total_correct += 1

                total_samples += 1

            except Exception as e:
                logger.debug(f"Sample evaluation error: {e}")
                continue

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.results.append(EvaluationResult(
            task_name="asof_qa",
            metric_name="accuracy",
            value=accuracy,
            unit="percent",
            model_type=model_type,
            timestamp=time.time(),
        ))

        logger.info(f"  As-of-QA: {accuracy:.4f} ({total_correct}/{total_samples})")
        return accuracy

    def test_causal_query(
        self,
        model_apply_fn,
        params: dict,
        dataloader: list[TaskSample],
        model_type: str = "ple_coded",
    ) -> float:
        logger.info(f"Running causal query benchmark ({model_type})")

        total_correct = 0
        total_samples = 0

        causal_samples = [s for s in dataloader if s.task_type == "causal"]

        for sample in causal_samples:
            input_ids = jnp.array(sample.input_ids).unsqueeze(0)

            try:
                logits = model_apply_fn(params, input_ids)

                if hasattr(logits, "logits"):
                    logits = logits.logits

                pred_token = int(jnp.argmax(logits[0, -1]))
                expected = sample.expected_tokens[0]

                if pred_token == expected:
                    total_correct += 1

                total_samples += 1

            except Exception as e:
                logger.debug(f"Sample evaluation error: {e}")
                continue

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.results.append(EvaluationResult(
            task_name="causal_query",
            metric_name="accuracy",
            value=accuracy,
            unit="percent",
            model_type=model_type,
            timestamp=time.time(),
        ))

        logger.info(f"  Causal query: {accuracy:.4f} ({total_correct}/{total_samples})")
        return accuracy

    def run_all_benchmarks(
        self,
        model_apply_fn,
        params: dict,
        model_type: str = "ple_coded",
    ) -> dict[str, float]:
        samples = self.data_generator.generate_dataloader(self.config)

        results = {}

        if self.config.test_staleness:
            results["staleness"] = self.test_staleness_detection(model_apply_fn, params, samples, model_type)

        if self.config.test_asof_qa:
            results["asof_qa"] = self.test_asof_qa(model_apply_fn, params, samples, model_type)

        if self.config.test_causal_query:
            results["causal_query"] = self.test_causal_query(model_apply_fn, params, samples, model_type)

        return results


class EdgeBenchmark:
    def measure_memory_footprint(self, params: dict) -> dict[str, float]:
        param_size = 0
        for v in params.values():
            if hasattr(v, 'shape'):
                param_size += v.size * v.dtype.itemsize
            elif isinstance(v, dict):
                for sv in v.values():
                    if hasattr(sv, 'shape'):
                        param_size += sv.size * sv.dtype.itemsize

        return {
            "param_mb": param_size / 1024 / 1024,
            "buffer_mb": 0.0,
            "total_mb": param_size / 1024 / 1024,
        }

    def measure_latency(
        self,
        model_apply_fn,
        params: dict,
        input_ids: jnp.ndarray,
        num_runs: int = 100,
    ) -> dict[str, float]:
        import time

        for _ in range(10):
            _ = model_apply_fn(params, input_ids)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model_apply_fn(params, input_ids)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        return {
            "latency_mean_ms": sum(latencies) / len(latencies),
            "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
            "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        }

    def benchmark_raspberry_pi(
        self,
        model_apply_fn,
        params: dict,
    ) -> dict:
        logger.info("Raspberry Pi benchmark (simulated)")

        memory = self.measure_memory_footprint(params)
        rng = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(rng, (1, 128), 0, 32000)
        latency = self.measure_latency(model_apply_fn, params, input_ids, num_runs=20)

        return {
            "target": "raspberry_pi",
            "memory_mb": memory["total_mb"],
            "latency_mean_ms": latency["latency_mean_ms"],
            "latency_p99_ms": latency["latency_p99_ms"],
            "note": "Simulated — actual deployment requires cross-compilation",
        }

    def benchmark_mobile(
        self,
        model_apply_fn,
        params: dict,
    ) -> dict:
        logger.info("Mobile benchmark (simulated)")

        memory = self.measure_memory_footprint(params)
        rng = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(rng, (1, 128), 0, 32000)
        latency = self.measure_latency(model_apply_fn, params, input_ids, num_runs=20)

        return {
            "target": "mobile",
            "memory_mb": memory["total_mb"],
            "latency_mean_ms": latency["latency_mean_ms"],
            "latency_p99_ms": latency["latency_p99_ms"],
            "note": "Simulated — actual deployment requires iOS/Android SDK",
        }


def evaluate_model(
    model_apply_fn,
    params: dict,
    config: Optional[TemporalBenchConfig] = None,
    model_type: str = "ple_coded",
) -> dict:
    if config is None:
        config = TemporalBenchConfig()

    bench = TemporalBench(config)
    edge = EdgeBenchmark()

    temporal_results = bench.run_all_benchmarks(model_apply_fn, params, model_type)
    memory_result = edge.measure_memory_footprint(params)

    return {
        "temporal_bench": temporal_results,
        "memory": memory_result,
    }


def compare_ple_coded_vs_baseline(
    ple_coded_apply_fn,
    ple_coded_params: dict,
    baseline_apply_fn,
    baseline_params: dict,
    config: Optional[TemporalBenchConfig] = None,
) -> dict:
    if config is None:
        config = TemporalBenchConfig()

    ple_results = evaluate_model(ple_coded_apply_fn, ple_coded_params, config, "ple_coded")
    baseline_results = evaluate_model(baseline_apply_fn, baseline_params, config, "baseline_q4")

    return {
        "ple_coded": ple_results,
        "baseline": baseline_results,
    }


def save_evaluation_results(results: dict, output_path: Path):
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return obj

    serializable = make_serializable(results)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")


@dataclass
class ComparisonReport:
    task_name: str
    ple_coded_score: float
    baseline_score: float
    delta: float
    delta_percent: float
    winner: str


class ModelComparator:
    def __init__(self, config: Optional[TemporalBenchConfig] = None):
        self.config = config or TemporalBenchConfig()
        self.bench = TemporalBench(self.config)

    def run_comparison(
        self,
        ple_coded_apply_fn,
        ple_coded_params: dict,
        baseline_apply_fn,
        baseline_params: dict,
        model_label: str = "baseline_q4",
    ) -> dict[str, ComparisonReport]:
        logger.info("Running model comparison...")

        ple_results = self.bench.run_all_benchmarks(ple_coded_apply_fn, ple_coded_params, "ple_coded")
        baseline_results = self.bench.run_all_benchmarks(baseline_apply_fn, baseline_params, model_label)

        reports = {}
        for task in ple_results:
            ple_score = ple_results[task]
            baseline_score = baseline_results.get(task, 0.0)
            delta = ple_score - baseline_score
            delta_pct = (delta / baseline_score * 100) if baseline_score > 0 else 0.0

            winner = "ple_coded" if ple_score > baseline_score else model_label
            if abs(delta) < 0.001:
                winner = "tie"

            reports[task] = ComparisonReport(
                task_name=task,
                ple_coded_score=ple_score,
                baseline_score=baseline_score,
                delta=delta,
                delta_percent=delta_pct,
                winner=winner,
            )

        return reports

    def print_comparison_report(self, reports: dict[str, ComparisonReport]):
        logger.info("=" * 70)
        logger.info("PLE-CODED VS BASELINE COMPARISON")
        logger.info("=" * 70)

        ple_wins = 0
        baseline_wins = 0
        ties = 0

        for task, report in reports.items():
            status = "WIN" if report.winner == "ple_coded" else ("LOSE" if report.winner != "tie" else "TIE")

            logger.info(f"{task.upper()}")
            logger.info(f"  PLE-Coded:     {report.ple_coded_score:.4f}")
            logger.info(f"  Baseline:      {report.baseline_score:.4f}")
            logger.info(f"  Delta:         {report.delta:+.4f} ({report.delta_percent:+.2f}%)")
            logger.info(f"  Status:        {status}")
            logger.info("")

            if report.winner == "ple_coded":
                ple_wins += 1
            elif report.winner == "tie":
                ties += 1
            else:
                baseline_wins += 1

        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"PLE-Coded wins:  {ple_wins}")
        logger.info(f"Baseline wins:   {baseline_wins}")
        logger.info(f"Ties:           {ties}")
        logger.info("=" * 70)


class TransformerEncoderLayer(nn.Module):
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


class MockModel(nn.Module):
    def setup(self):
        self.embedding = nn.Embed(32000, 2048)
        self.layers = [
            TransformerEncoderLayer(d_model=2048, nhead=8)
            for _ in range(6)
        ]
        self.lm_head = nn.Dense(32000, use_bias=False)

    def __call__(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return type('Output', (), {'logits': self.lm_head(x)})()


def create_mock_baseline_model():
    model = MockModel()
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    return model, variables["params"]
