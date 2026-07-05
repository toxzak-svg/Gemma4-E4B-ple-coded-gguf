#!/usr/bin/env python3
"""Launch a Colab-friendly Gradio runner for PLE-Coded GGUF."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def run_command(args: Iterable[str]) -> str:
    proc = subprocess.run(
        list(args),
        cwd=ROOT,
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout.strip()
    status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    return f"$ {shlex.join(args)}\n\n[{status}]\n{output}"


def make_app():
    import gradio as gr

    def mock_profile() -> str:
        return run_command([sys.executable, "-m", "profiling.mock_profiler"])

    def quick_profile() -> str:
        return run_command([sys.executable, "profiling/quick_profile.py"])

    def full_pipeline(
        model_name: str,
        num_samples: int,
        seq_len: int,
        adapter_epochs: int,
        use_mock: bool,
        no_eval: bool,
    ) -> str:
        cmd = [
            sys.executable,
            "-m",
            "profiling.pipeline",
            "--model-source",
            "huggingface",
            "--model-name",
            model_name.strip() or "google/gemma-4-E2B-it",
            "--num-samples",
            str(num_samples),
            "--seq-len",
            str(seq_len),
            "--adapter-epochs",
            str(adapter_epochs),
        ]
        if use_mock:
            cmd.append("--use-mock")
        if no_eval:
            cmd.append("--no-eval")
        return run_command(cmd)

    with gr.Blocks(title="PLE-Coded GGUF Colab Runner") as app:
        gr.Markdown("# PLE-Coded GGUF Colab Runner")
        gr.Markdown(
            "Run the existing profiling and pipeline commands from a Colab GPU runtime. "
            "Set `HF_TOKEN` and, for ngrok, `NGROK_AUTHTOKEN` in Colab secrets or environment variables."
        )

        with gr.Row():
            mock_button = gr.Button("Run mock profiling")
            quick_button = gr.Button("Run quick real profile")

        with gr.Row():
            model_name = gr.Textbox(
                label="Hugging Face model",
                value="google/gemma-4-E2B-it",
            )
            num_samples = gr.Number(label="Samples", value=64, precision=0)
            seq_len = gr.Number(label="Sequence length", value=128, precision=0)
            adapter_epochs = gr.Number(label="Adapter epochs", value=1, precision=0)

        with gr.Row():
            use_mock = gr.Checkbox(label="Use mock profiling in full pipeline", value=False)
            no_eval = gr.Checkbox(label="Skip evaluation", value=False)
            pipeline_button = gr.Button("Run full pipeline", variant="primary")

        output = gr.Textbox(label="Command output", lines=30, max_lines=60)

        mock_button.click(mock_profile, outputs=output)
        quick_button.click(quick_profile, outputs=output)
        pipeline_button.click(
            full_pipeline,
            inputs=[model_name, num_samples, seq_len, adapter_epochs, use_mock, no_eval],
            outputs=output,
        )

    return app


def open_ngrok_tunnel(port: int) -> str:
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError("NGROK_AUTHTOKEN is required for --tunnel ngrok")

    from pyngrok import ngrok

    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(port, bind_tls=True)
    return str(tunnel.public_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Colab tunnel UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tunnel", choices=["ngrok", "gradio", "none"], default="ngrok")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (ROOT / "profiling").exists():
        raise SystemExit(f"Expected repo root at {ROOT}")

    share = args.tunnel == "gradio"
    public_url = None

    if args.tunnel == "ngrok":
        public_url = open_ngrok_tunnel(args.port)
        print(f"ngrok tunnel: {public_url}", flush=True)

    app = make_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
