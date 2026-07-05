# Colab Tunnel Runbook

This setup runs the existing `profiling.pipeline` CLI from a Colab GPU runtime and exposes a small Gradio control panel through either ngrok or Gradio's share tunnel.

## 1. Start a Colab GPU Runtime

Use `Runtime -> Change runtime type -> T4/A100 GPU`.

Clone this repo in the first cell:

```python
!git clone https://github.com/YOUR_USER/ple-coded-gguf.git
%cd ple-coded-gguf
```

If you upload the repo zip instead, change into the uploaded `ple-coded-gguf` directory before continuing.

## 2. Install Dependencies

```python
%pip install -r colab/requirements-colab.txt
```

Restart the runtime if Colab asks after installing JAX or Torch.

## 3. Set Tokens Without Committing Them

Use the Colab secrets sidebar when possible:

- `HF_TOKEN`: Hugging Face token with access to gated Gemma weights.
- `NGROK_AUTHTOKEN`: ngrok token, only needed for `--tunnel ngrok`.

Or set them in a temporary cell:

```python
import os, getpass
os.environ["HF_TOKEN"] = getpass.getpass("HF token: ")
os.environ["NGROK_AUTHTOKEN"] = getpass.getpass("ngrok token: ")
```

## 4. Launch the Tunnel Control Panel

For ngrok:

```python
!python colab/run_colab_tunnel.py --tunnel ngrok --port 7860
```

For Gradio share links without ngrok:

```python
!python colab/run_colab_tunnel.py --tunnel gradio --port 7860
```

The cell prints the public URL. Open it and use the buttons to run:

- mock profiling smoke
- quick real-model profiling
- full pipeline

## Direct Commands

Mock smoke:

```python
!python -m profiling.mock_profiler
```

Small real-model profile:

```python
!python profiling/quick_profile.py
```

Full pipeline:

```python
!python -m profiling.pipeline --model-source huggingface --model-name google/gemma-4-E2B-it --num-samples 64 --seq-len 128 --adapter-epochs 1
```

## Notes

- Do not paste tokens into tracked files.
- Use mock profiling first to verify imports before spending GPU time.
- Real Gemma access depends on your Hugging Face account approval and token scope.
