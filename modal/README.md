# Modal Qwen deployment

Optional cloud inference for Local LLM using Qwen3.8-27B on Modal. The Rust
controller manages deployment and shutdown; the Rust loopback gateway handles
Modal cold starts and passes through streamed responses. CUDA inference runs
directly in llama.cpp. The Modal deployment uses Python for its SDK lifecycle shim.

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

| Profile | GPU | Weight quant | KV cache | Context | Benchmark-era minimum active rate |
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
- IQ4 model ID: `qwen38-hauhau-iq4xs` (only use with the IQ4 profile)
- Existing VansRouter: `http://127.0.0.1:20128/v1`

| Active profile | VansRouter model | 128K combo |
| --- | --- | --- |
| `q8` | `mq8/qwen38-hauhau-q8kp` | `qwen-modal-q8-128k` |
| `iq4` | `mq8/qwen38-hauhau-iq4xs` | `qwen-modal-iq4-128k` |

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

The controller records the verified active profile in a local state file.
The gateway refuses inference with a mismatched model ID or stopped state,
preventing an accidental request on the more expensive profile. If you change
the deployment outside the controller, run `qwen-modal start` for the desired
profile to reconcile the local state. Use the matching router model or combo.

The gateway only retries HTTP 503 **before** the upstream responds successfully.
If a connection breaks after streaming has begun, the client gets the stream
error rather than silently replaying a potentially expensive generation.
Modal may still be unable to allocate a GPU when supply is exhausted.
If the router times out waiting for cold capacity, run `qwen-modal start` and
wait for its readiness confirmation before sending the next request.

## Build and install

Run the commands from the repository root on a host with the configured
`local-llm` Modal profile and model Volume.

```sh
cargo build --release --locked --manifest-path modal/qwen-control/Cargo.toml
cargo build --release --locked --manifest-path modal/qwen-gateway/Cargo.toml
install -m 755 modal/qwen-control/target/release/qwen-modal ~/.local/bin/qwen-modal
install -m 755 modal/qwen-gateway/target/release/qwen-modal-gateway ~/.local/bin/qwen-modal-gateway
install -m 644 modal/qwen-modal-gateway.service ~/.config/systemd/user/qwen-modal-gateway.service
systemctl --user daemon-reload
systemctl --user enable --now qwen-modal-gateway.service
```

The controller looks for the checkout and its virtual environment in
`~/local-llm`. Set `LOCAL_LLM_ROOT` to the absolute checkout path to use another
directory. For a fresh checkout, install `modal==1.6.4` in `.venv-modal`
and configure the `local-llm` profile using Modal's CLI. The server image is
pinned to the image that passed the Q8 benchmark in this workspace. A new
Modal workspace will need its own image and model volume.

The root `modal_qwen.py` symlink lets previously installed controllers open
the deployment script in its new location.

Check before installing a new version:

```sh
cargo test --locked --manifest-path modal/qwen-control/Cargo.toml
cargo test --locked --manifest-path modal/qwen-gateway/Cargo.toml
.venv-modal/bin/python -m py_compile modal/modal_qwen.py
```

`modal_q8_bench.py` records the isolated 127K-token benchmark method. It is not
part of the running service and should only be run intentionally: it consumes
Modal credits. To reproduce the embedded MTP run:

```sh
QWEN_BENCH_MTP=1 .venv-modal/bin/modal run --profile local-llm modal/modal_q8_bench.py::benchmark
```

## Benchmark results

On the tested L40S, the Q8 baseline generated 22.04 tokens/s on a short
prompt. Embedded MTP increased that to 35.12 tokens/s, using about 37.8 GiB of
45.0 GiB VRAM. It recalled the test code correctly from a 127044-token
synthetic prompt (127.3 s prefill, 30.27 tokens/s for the brief answer).
These are fixture results, not guarantees of real coding-agent throughput.

End-to-end verification through VansRouter and the Rust gateway also completed
a cold-start Anthropic SSE request (HTTP 200, 46.2 s until first byte), then a
cold-start OpenAI SSE request containing 127044 prompt tokens. The latter
returned the correct recall code, completed with `[DONE]`, and took 178.6 s
including GPU scheduling and model load. Cold start times vary with capacity.
The IQ4 fallback also completed a cold streaming request through the gateway.
After switching back to Q8, an Anthropic tool-use stream returned the expected
tool name and arguments, the Q8 model ID, and `stop_reason: tool_use`; it took
2.85 s after readiness. No tool was executed during the test.
