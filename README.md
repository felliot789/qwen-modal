# Local LLM

A Linux homelab toolkit for running GGUF models with llama.cpp, monitoring the
host, and connecting apps to an OpenAI-compatible API.

## Quick start

Use Python 3.10+ on Linux. To launch models, install a compatible
`llama-server` and put your GGUF files under `~/models`.

```sh
python3 monitor/server.py
```

Open the dashboard, select a GGUF, and set its context size, CPU threads, and
GPU layers. Once the model is ready, its chat UI is on port `8791` and API
clients can use `http://127.0.0.1:8791/v1`. [Open the local dashboard](http://127.0.0.1:8790/)

## Project layout

- **`monitor/`** contains the hardware dashboard, model launcher, tests, and an
  optional proxy for model discovery and chat completion. The dashboard tracks
  CPU, AMD GPU, RAM, and available temperature, fan, and power sensors.
  [Dashboard and API guide](monitor/README.md)
- **`modal/`** contains the optional cloud deployment, Rust controller, streaming
  gateway, and benchmark. Use these tools when you want to run Qwen on Modal.
  [Modal setup and benchmarks](modal/README.md)

## Checks

```sh
python3 -m unittest discover -s monitor
node --check monitor/app.js
```

The dashboard guide covers local, SSH, and Tailscale access. The Modal guide
includes separate Rust build and test commands.
