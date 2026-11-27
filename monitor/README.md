# Dashboard and API

Run the monitor on the Linux machine whose hardware you want to see. Python
3.10+ serves the dashboard without third-party Python packages. Install
`llama-server` to use the GGUF runner. The monitor checks
`~/local-llm/llama.cpp/build/bin/llama-server`, then
`~/llama.cpp/build/bin/llama-server`, and then `PATH`. Use
`--llama-server /path/to/llama-server` to select a different build.

## Run locally

From the repository root:

```sh
python3 monitor/server.py
```

The model picker scans `~/models` and supports complete split GGUF models.
Use `--models-dir /path/to/models` to include another directory. Set GPU layers
to zero for CPU-only inference. The runner controls one model process and
stops it on a normal dashboard shutdown. It also lists other local
`llama-server` processes for visibility.

Select Ornith Q5_K_M to use the Vulkan presets when the llama.cpp build supports
the shared MoE expert cache. The 16K speed preset requests a 2,352 MiB cache.
The 64K preset requests 2,080 MiB. Both were tuned for the headless RX 6500 XT
and used about 4,000 MiB of VRAM under load.

The hardware dashboard samples every two seconds and holds two minutes of
utilization history in memory. AMD GPU readings come from the driver. CPU and
DRAM power readings depend on readable Linux RAPL counters. Wall power
requires a separate meter.

## Addresses and access

| Access | Dashboard | Runner API base URL |
| --- | --- | --- |
| Local | `http://127.0.0.1:8790` | `http://127.0.0.1:8791/v1` |
| Tailscale HTTPS | `https://<MagicDNS hostname>:8790` | `https://<MagicDNS hostname>:8791/v1` |
| Direct tailnet IP | `http://<Tailscale IPv4>:8789` | `http://<Tailscale IPv4>:8788/v1` |

Both local listeners bind to loopback. The runner API and the llama.cpp chat
UI are available once a model is ready. Use `--port` and `--model-port` if
the defaults are occupied.

### SSH

Forward both local ports from your client, then use the local addresses in
the table.

```sh
ssh -N -L 8790:127.0.0.1:8790 -L 8791:127.0.0.1:8791 user@host
```

### Tailscale

For HTTPS, start the monitor with `--tailnet-host <your MagicDNS hostname>`.

```sh
tailscale serve --bg --https=8790 http://127.0.0.1:8790
tailscale serve --bg --https=8791 http://127.0.0.1:8790
```

For direct IP access, start it with `--tailnet-ip "$(tailscale ip -4)"`. This
uses HTTP inside the encrypted tailnet and avoids a hostname lookup.

```sh
tailscale serve --bg --tcp=8789 tcp://127.0.0.1:8790
tailscale serve --bg --tcp=8788 tcp://127.0.0.1:8790
```

Both Tailscale paths go through the monitor, which routes model requests to
the local runner. If you customize the local ports, set the Serve target to
the dashboard port and keep each external port aligned with the monitor's
HTTPS or direct IP configuration.

## Connect a client

Set the client's OpenAI-compatible base URL to one of the runner addresses
above. Use the model ID returned by `GET /v1/models`, then send messages to
`POST /v1/chat/completions`.

To show a separate no-key chat service in the dashboard, start the monitor
with `--api-base-url <base URL ending in /v1>`. The dashboard checks its model
listing and displays the address, model ID, and availability. The separate
service controls its own model lifecycle.

### Public HTTPS proxy

The optional [Nginx template](public-api.nginx.conf) serves model discovery
and chat completions on loopback port `8792`. Substitute `LLAMA_UPSTREAM`
with the serving llama.cpp host and port and run Nginx using the rendered
configuration. The proxy accepts one concurrent generation with requests up
to 4 MiB.

With Tailscale Funnel enabled, mount the proxy under a dedicated path:

```sh
tailscale funnel --bg --https=443 --set-path=/local-llm http://127.0.0.1:8792
```

Clients use `https://<Funnel hostname>/local-llm/v1` as their base URL. Funnel
makes this no-key endpoint reachable from the internet. Provide the same
base URL with `--api-base-url` to display it in the dashboard.

### Intel CPU and DRAM power

When Linux limits RAPL energy counters to root, install the dedicated sampler from a trusted checkout. It publishes only rounded two-second watt averages to `/run/local-llm-rapl/power.json`. The dashboard runs as your regular user and ignores samples older than ten seconds.
Set the service group to a private group belonging to the dashboard account. The
runtime directory and snapshot grant that group read access.

```sh
sudo install -D -o root -g root -m 0644 monitor/rapl-sampler.py /usr/local/libexec/local-llm-rapl.py
sudo install -o root -g root -m 0644 monitor/local-llm-rapl.service /etc/systemd/system/local-llm-rapl.service
sudo install -d -o root -g root -m 0755 /etc/systemd/system/local-llm-rapl.service.d
printf '[Service]\nGroup=%s\n' "$(id -gn)" |
  sudo tee /etc/systemd/system/local-llm-rapl.service.d/access.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now local-llm-rapl.service
```
