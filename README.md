# Qwen on Modal

Single-user, authenticated Qwen3.8-27B inference for the homelab. The Rust
controller manages deployment and shutdown; the Rust loopback gateway handles
Modal cold starts and passes through streamed responses. CUDA inference runs
directly in llama.cpp. Python is only the Modal SDK's deployment/lifecycle shim.

## Daily use

```sh
qwen-modal start       # deploy/wake Q8 and wait until the correct model is ready
qwen-modal status      # inspect deployment and active container count without waking it
qwen-modal stop        # disable the deployment, even if a request is in progress
```

To switch to the smaller profile, stop first, then `qwen-modal start iq4`.
Return to Q8 using `qwen-modal stop` followed by `qwen-modal start`.
Both model files are retained in the authenticated Modal Volume. No redownload
is required.

| Profile | GPU | Weight quant | KV cache | Context | Estimated minimum active rate |
| --- | --- | --- | --- | ---: | ---: |
| `q8` (default) | L40S | Q8_K_P + embedded MTP | F16 | 131072 | $2.01/h |
| `iq4` | L4 | IQ4_XS | Q4_0 | 131072 | $0.86/h |

Each profile requests 1 physical CPU core (up to 2) and 2 GiB RAM. The memory
request is not a hard cap. Actual CPU/RAM use above the request can increase
billing. There is never more than one GPU container, and the idle scale-down
window is 30 seconds. The gateway's local health check and CLI `status` do not
wake the GPU. When deployed but idle, an authenticated inference request can
wake it. If you want to prevent further requests from waking it, use `stop`.
Pricing: [Modal](https://modal.com/pricing). See [resource billing](https://modal.com/docs/guide/resources).

## API routing

- Modal endpoint: `https://makunku3--qwen38-27b-server.us-east.modal.direct/v1`
- Rust gateway (local only): `http://127.0.0.1:20129/v1`
- Gateway liveness: `http://127.0.0.1:20129/healthz` (no GPU call)
- Q8 model ID: `qwen38-hauhau-q8kp`
- IQ4 model ID: `qwen38-hauhau-iq4xs` (also accepted on Q8 for legacy callers)
- Existing VansRouter: `http://127.0.0.1:20128/v1`, model `mq8/qwen38-hauhau-q8kp`
- VansRouter combo advertising the 128K window: `qwen-modal-q8-128k`

Use the gateway as the upstream to VansRouter. Pass the Modal proxy bearer
token as the upstream API key. The gateway does not store or log that token;
it forwards authentication to the fixed Modal endpoint. Never put a key in
this repository. The existing separate managed BF16 deployment is independent.

Claude Code uses the existing local router URL and credentials. Override the
model and context for one session without editing its global settings:

```sh
claude --model mq8/qwen38-hauhau-q8kp --settings '{"env":{"CLAUDE_CODE_MAX_CONTEXT_TOKENS":"131072"}}'
```

Set the client context window to `131072`, including prompt, history, and
generation. Reserve room for the response when choosing the prompt length.

The gateway only retries HTTP 503 **before** the upstream responds successfully.
If a connection breaks after streaming has begun, the client gets the stream
error rather than silently replaying a potentially expensive generation.
Modal may still be unable to allocate a GPU when supply is exhausted.

## Build and install

On the homelab, with the configured `local-llm` Modal CLI profile and the
existing model Volume:

```sh
cargo build --release --locked --manifest-path qwen-control/Cargo.toml
cargo build --release --locked --manifest-path qwen-gateway/Cargo.toml
install -m 755 qwen-control/target/release/qwen-modal ~/.local/bin/qwen-modal
install -m 755 qwen-gateway/target/release/qwen-modal-gateway ~/.local/bin/qwen-modal-gateway
install -m 644 qwen-modal-gateway.service ~/.config/systemd/user/qwen-modal-gateway.service
systemctl --user daemon-reload
systemctl --user enable --now qwen-modal-gateway.service
```

The controller expects this checkout and its Modal virtual environment in
`~/local-llm`. For a fresh checkout, install `modal==1.6.4` in `.venv-modal`
and configure the `local-llm` profile using Modal's CLI. The server image is
pinned to the image that passed the Q8 benchmark in this workspace. A new
Modal workspace will need its own image and model volume.

Check before installing a new version:

```sh
cargo test --locked --manifest-path qwen-control/Cargo.toml
cargo test --locked --manifest-path qwen-gateway/Cargo.toml
.venv-modal/bin/python -m py_compile modal_qwen.py
```

`modal_q8_bench.py` records the isolated 127K-token benchmark method. It is not
part of the running service and should only be run intentionally: it consumes
Modal credits. To reproduce the embedded MTP run:

```sh
QWEN_BENCH_MTP=1 .venv-modal/bin/modal run --profile local-llm modal_q8_bench.py::benchmark
```

On the tested L40S, the Q8 baseline generated 22.04 tokens/s on a short
prompt. Embedded MTP increased that to 35.12 tokens/s, using about 37.8 GiB of
45.0 GiB VRAM. It recalled the test code correctly from a 127044-token
synthetic prompt (127.3 s prefill, 30.27 tokens/s for the brief answer).
These are fixture results, not guarantees of real coding-agent throughput.
