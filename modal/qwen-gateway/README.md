# Qwen Modal Gateway

A loopback-only Rust streaming proxy for the private Modal Qwen app. The
gateway itself does not allocate a GPU. It waits through Modal's cold-start
HTTP 503 responses (up to a 180-second retry window), then forwards the actual
response and its stream without waiting for the full completion. A response
already being streamed is never retried, and a disconnected caller cancels
pending retries.

```sh
cargo test --locked
cargo build --locked --release
./target/release/qwen-modal-gateway
```

- Local OpenAI-compatible base URL: `http://127.0.0.1:20129/v1`
- Local liveness check (does not wake Modal): `curl -fsS http://127.0.0.1:20129/healthz`
- Pass the existing Modal proxy-token bearer or `Modal-Key` / `Modal-Secret`
  headers to the gateway as you would to the Modal endpoint. No credentials
  are stored or logged by the gateway.
- The `qwen-modal` controller writes the active variant to
  `~/.local/state/qwen-modal/variant`. Missing or stopped state rejects requests
  locally; only the selected variant's model ID is accepted for inference.
  Changing or stopping the variant also cancels pending cold-start retries.
- Normal requests, authentication, and model availability are handled by
  Modal and llama.cpp. Non-503 errors are returned immediately. Accepted
  inference may run longer than the cold-start retry window.

Request bodies are limited to 16 MiB; response bodies are streamed. The only
upstream is the Qwen app on Modal, with TLS, no redirects, and no process
proxy configuration. The gateway does not create, pin, or stop cloud
containers; use `qwen-modal` to manage the deployment.
