"""Local web dashboard and llama-server runner. Linux and Python 3.10+."""

import argparse
from collections import deque
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address, IPv4Network
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
from threading import Event, Lock, Thread
import time
from urllib.parse import urlsplit

from telemetry import INTERVAL, Telemetry


STATIC = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
LOOPBACK = "127.0.0.1"
TAILNET_HTTP_DASHBOARD = 8789
TAILNET_HTTP_MODEL = 8788
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf", re.I)


def model_catalog(roots):
    models = {}
    for root in roots:
        files = {}
        for candidate in root.rglob("*"):
            if candidate.suffix.lower() != ".gguf" or candidate.name.lower().startswith("mmproj"):
                continue
            try:
                if candidate.is_file() and candidate.resolve().is_relative_to(root):
                    files[candidate] = candidate.stat().st_size
            except OSError:
                continue
        for path, size in files.items():
            name = path.stem
            match = SHARD.fullmatch(path.name)
            if match:
                name, index, count = match.groups()
                if index != "00001":
                    continue
                siblings = [
                    path.with_name(f"{name}-{part:05d}-of-{count}{path.suffix}")
                    for part in range(1, int(count) + 1)
                ]
                if not siblings or any(part not in files for part in siblings):
                    continue
                size = sum(files[part] for part in siblings)
            if size > 0:
                models[str(path)] = {
                    "id": str(path), "name": name, "bytes": size,
                    "folder": str(path.parent.relative_to(root)),
                }
    return sorted(models.values(), key=lambda model: (model["name"].casefold(), model["id"]))


ORNITH_Q5 = "Ornith-1.5-35B-A3B-CRACK-Q5_K_M.gguf"
PRESETS = {
    "ornith-q5-16k": (16384, 2352),
    "ornith-q5-64k": (65536, 2080),
}


def find_llama(configured):
    if configured:
        return shutil.which(os.path.expanduser(configured))
    for root in (Path.home() / "local-llm", Path.home()):
        built = shutil.which(str(root / "llama.cpp/build/bin/llama-server"))
        if built:
            return built
    return shutil.which("llama-server")


def collect_logs(stream, lines):
    with stream:
        while line := stream.readline(4096):
            lines.append(line)


def active_api_model(base_url):
    url = urlsplit(base_url)
    connect = HTTPSConnection if url.scheme == "https" else HTTPConnection
    connection = connect(url.hostname, url.port, timeout=0.8)
    try:
        connection.request("GET", url.path + "/models")
        response = connection.getresponse()
        if response.status == 200:
            payload = json.loads(response.read(65536))
            models = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(models, list):
                for entry in models:
                    model_id = entry.get("id") if isinstance(entry, dict) else None
                    if isinstance(model_id, str) and model_id:
                        return model_id
    except (OSError, HTTPException, ValueError):
        return None
    finally:
        connection.close()


class Runner:
    def __init__(self, binary, port, roots):
        self.binary = binary
        self.preset_supported = False
        if binary:
            try:
                help_text = subprocess.run(
                    [binary, "--help"], capture_output=True, timeout=5, check=True,
                ).stdout
                self.preset_supported = b"--moe-cache-mib" in help_text
            except (OSError, subprocess.SubprocessError):
                pass
        self.port = port
        self.roots = roots
        self.url = f"http://{LOOPBACK}:{port}"
        self.lock = Lock()
        self.process = None
        self.model = None
        self.state = "idle"
        self.exit_code = None
        self.stop_deadline = None
        self.closed = False
        self.logs = deque(maxlen=160)

    def start(self, params):
        model_id = params.get("model")
        model = next((m for m in model_catalog(self.roots) if m["id"] == model_id), None)
        if model is None:
            raise ValueError("Choose a GGUF from the model library.")
        preset = params.get("preset", "")
        if not isinstance(preset, str) or (preset and preset not in PRESETS):
            raise ValueError("Choose a supported launch preset.")
        if preset and (not self.preset_supported or Path(model_id).name != ORNITH_Q5):
            raise ValueError("This preset requires Ornith Q5_K_M and the expert-cache llama.cpp build.")
        context = params.get("context")
        threads = params.get("threads")
        layers = params.get("layers")
        if preset:
            context, cache_mib = PRESETS[preset]
            threads, layers = 4, 99
        if type(context) is not int or not 256 <= context <= 262144:
            raise ValueError("Enter a context size between 256 and 262144.")
        if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Enter a thread count within the available CPU threads.")
        if type(layers) is not int or not 0 <= layers <= 999:
            raise ValueError("Enter a GPU layer count between 0 and 999.")
        if not self.binary:
            raise ValueError("llama-server was not found. Start the monitor with --llama-server.")
        with self.lock:
            if self.closed or (self.process and self.process.poll() is None):
                raise ValueError("The runner is already busy.")
            with socket.socket() as probe:
                try:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind((LOOPBACK, self.port))
                except OSError as exc:
                    raise ValueError(f"Model port {self.port} is already in use.") from exc
            command = [
                self.binary, "--model", model_id, "--host", LOOPBACK, "--port", str(self.port),
                "--ctx-size", str(context), "--threads", str(threads), "--parallel", "1",
                "--gpu-layers", str(layers), "--fit", "off", "--jinja",
            ]
            if preset:
                command.extend([
                    "--device", "Vulkan0", "--flash-attn", "on",
                    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
                    "--n-cpu-moe", "40", "--no-op-offload", "--load-mode", "mmap",
                    "--no-host", "--no-repack", "--batch-size", "2048", "--ubatch-size", "128",
                    "--cache-ram", "0", "--no-mmproj", "--offline", "--ctx-checkpoints", "0",
                    "--no-context-shift", "--no-cache-idle-slots", "--spec-type", "none",
                    "--reasoning", "off", "--perf", "--moe-cache-mib", str(cache_mib),
                    "--moe-cache-cpu-share", "0.75",
                ])
            elif layers == 0:
                command.extend(["--device", "none", "--no-op-offload"])
            env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
            try:
                child = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, env=env, start_new_session=True,
                )
            except OSError as exc:
                raise ValueError(f"Could not launch llama-server. {exc.strerror}.") from exc
            self.process = child
            self.model = model["name"]
            self.state = "loading"
            self.exit_code = None
            self.stop_deadline = None
            # A prior model's reader may still be draining its pipe after an exit.
            self.logs = deque(maxlen=160)
            Thread(target=collect_logs, args=(child.stdout, self.logs), daemon=True).start()

    def refresh(self):
        with self.lock:
            if self.process is None:
                return
            code = self.process.poll()
            if code is not None:
                self.exit_code = code
                self.state = "idle" if self.state == "stopping" else "failed"
                self.process = None
            elif self.state == "stopping":
                if time.monotonic() >= self.stop_deadline:
                    self.process.kill()
            else:
                connection = HTTPConnection(LOOPBACK, self.port, timeout=0.35)
                try:
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    data = json.loads(response.read(1024)) if response.status == 200 else None
                    ready = isinstance(data, dict) and data.get("status") == "ok"
                    self.state = "ready" if ready else "loading"
                except (OSError, ValueError, HTTPException):
                    self.state = "loading"
                finally:
                    connection.close()

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None and self.state != "stopping":
                self.state = "stopping"
                self.process.terminate()
                self.stop_deadline = time.monotonic() + 4

    def close(self):
        with self.lock:
            self.closed = True
        self.stop()
        with self.lock:
            if self.process:
                try:
                    self.process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

    def status(self):
        with self.lock:
            return {
                "state": self.state, "model": self.model,
                "pid": self.process.pid if self.process else None, "exit_code": self.exit_code,
                "url": self.url, "binary": bool(self.binary), "preset_supported": self.preset_supported,
                "logs": b"".join(self.logs.copy())[-40000:].decode(errors="replace"),
            }


class Monitor(ThreadingHTTPServer):
    def __init__(self, port, runner, tailnet_host=None, tailnet_ip=None, api_base_url=None):
        super().__init__((LOOPBACK, port), Handler)
        self.runner = runner
        self.origins = {f"{host}:{port}": f"http://{host}:{port}" for host in (LOOPBACK, "localhost")}
        self.runtime_urls = {}
        self.model_hosts = set()
        for scheme, address, dashboard_port, model_port in (
            ("https", tailnet_host, port, runner.port),
            ("http", tailnet_ip, TAILNET_HTTP_DASHBOARD, TAILNET_HTTP_MODEL),
        ):
            if address:
                dashboard = f"{address}:{dashboard_port}"
                model = f"{address}:{model_port}"
                self.origins[dashboard] = f"{scheme}://{dashboard}"
                self.origins[model] = f"{scheme}://{model}"
                self.runtime_urls[dashboard] = self.origins[model]
                self.model_hosts.add(model)
        self.telemetry = Telemetry()
        self.snapshot = self.telemetry.sample()
        self.stopped = Event()
        self.chat_api = {"base_url": api_base_url, "model": None, "checked": False} if api_base_url else None

    def sample_forever(self):
        next_api_check = 0
        while not self.stopped.wait(INTERVAL):
            self.snapshot = self.telemetry.sample()
            self.runner.refresh()
            if self.chat_api and time.monotonic() >= next_api_check:
                base_url = self.chat_api["base_url"]
                self.chat_api = {"base_url": base_url, "model": active_api_model(base_url), "checked": True}
                next_api_check = time.monotonic() + 15


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_):
        pass

    def reply(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def json(self, data, status=200):
        self.reply(json.dumps(data).encode(), "application/json", status)

    def check_origin(self):
        origin = self.server.origins.get(self.headers.get("Host"))
        if origin is None:
            self.json({"error": "Use the configured dashboard address."}, 403)
            return False
        if origin.startswith("https://") and self.headers.get("X-Forwarded-Proto") != "https":
            self.json({"error": "Use Tailscale Serve over HTTPS."}, 403)
            return False
        if self.headers.get("Origin") not in (None, origin):
            self.json({"error": "The request must come from the same origin."}, 403)
            return False
        return True

    def runner_status(self):
        status = self.server.runner.status()
        status["url"] = self.server.runtime_urls.get(self.headers["Host"], status["url"])
        return status

    def proxy_model(self):
        # Keep llama-server on loopback and apply the same origin checks to its UI and API.
        if self.headers.get("Transfer-Encoding"):
            self.json({"error": "Send the request with a Content-Length."}, 400)
            return
        length = self.headers.get("Content-Length", "0")
        if not length.isascii() or not length.isdecimal() or len(length) > 15:
            self.json({"error": "Invalid Content-Length."}, 400)
            return
        upstream = HTTPConnection(LOOPBACK, self.server.runner.port, timeout=300)
        try:
            upstream.putrequest(self.command, self.path, skip_accept_encoding=True)
            for name in ("Content-Type", "Authorization", "X-Api-Key"):
                if name in self.headers:
                    upstream.putheader(name, self.headers[name])
            if self.command == "POST":
                upstream.putheader("Content-Length", length)
            upstream.endheaders()
            remaining = int(length) if self.command == "POST" else 0
            while remaining:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    upstream.close()
                    return
                upstream.send(chunk)
                remaining -= len(chunk)
            response = upstream.getresponse()
        except (OSError, HTTPException):
            upstream.close()
            self.json({"error": "The model server is unavailable. Check the GGUF runner."}, 503)
            return
        try:
            self.send_response(response.status)
            self.send_header("Connection", "close")
            self.send_header("Referrer-Policy", "no-referrer")
            for name in ("Content-Type", "Content-Encoding", "Cache-Control", "Location"):
                if name in response.headers:
                    self.send_header(name, response.headers[name])
            self.end_headers()
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
        except (OSError, HTTPException):
            pass
        finally:
            upstream.close()

    def do_GET(self):
        if not self.check_origin():
            return
        if self.headers["Host"] in self.server.model_hosts:
            self.proxy_model()
        elif self.path == "/api/live":
            self.json({**self.server.snapshot, "runner": self.runner_status(), "chat_api": self.server.chat_api})
        elif self.path == "/api/models":
            self.json(model_catalog(self.server.runner.roots))
        elif self.path in ASSETS:
            filename, content_type = ASSETS[self.path]
            self.reply((STATIC / filename).read_bytes(), content_type)
        else:
            self.json({"error": "Page not found."}, 404)

    def do_POST(self):
        if not self.check_origin():
            return
        if self.headers["Host"] in self.server.model_hosts:
            self.proxy_model()
            return
        if self.path not in ("/api/start", "/api/stop"):
            self.json({"error": "Page not found."}, 404)
            return
        if self.headers.get_content_type() != "application/json":
            self.json({"error": "Use an application/json request."}, 415)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 8192:
                raise ValueError
            params = json.loads(self.rfile.read(size))
            if not isinstance(params, dict):
                raise ValueError
        except (ValueError, TimeoutError):
            self.json({"error": "Invalid request body."}, 400)
            return

        runner = self.server.runner
        try:
            if self.path == "/api/start":
                runner.start(params)
            else:
                runner.stop()
        except ValueError as exc:
            self.json({"error": str(exc)}, 400)
            return
        self.json(self.runner_status())

    def do_HEAD(self):
        if not self.check_origin():
            return
        if self.headers["Host"] in self.server.model_hosts:
            self.proxy_model()
        else:
            self.send_error(405)


def port_number(value):
    port = int(value)
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("choose a port between 1024 and 65535")
    return port


def api_base_url(value):
    try:
        url = urlsplit(value.rstrip("/"))
        if (
            url.scheme in ("http", "https") and url.hostname and url.username is None
            and url.path.endswith("/v1") and not url.query and not url.fragment
            and url.port != 0
        ):
            return url.geturl()
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("enter an HTTP or HTTPS base URL ending in /v1 with no credentials")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=8790)
    parser.add_argument("--model-port", type=port_number, default=8791)
    parser.add_argument("--models-dir", action="append", default=[], type=Path,
                        help="additional GGUF directory, repeatable")
    parser.add_argument("--llama-server", help="path to the llama-server executable")
    parser.add_argument("--tailnet-host", help="MagicDNS hostname for HTTPS via Tailscale Serve")
    parser.add_argument("--tailnet-ip", type=IPv4Address,
                        help="Tailscale IPv4 for direct access on TCP ports 8789 and 8788")
    parser.add_argument("--api-base-url", type=api_base_url,
                        help="URL for an existing no-key OpenAI-compatible server to display on the dashboard")
    args = parser.parse_args()
    if args.model_port == args.port:
        parser.error("the dashboard and model ports must be different")
    if args.tailnet_host:
        args.tailnet_host = args.tailnet_host.rstrip(".").lower()
        if not re.fullmatch(r"(?:[a-z0-9-]+\.)+ts\.net", args.tailnet_host):
            parser.error("--tailnet-host must be a .ts.net DNS hostname")
    if args.tailnet_ip:
        if args.tailnet_ip not in IPv4Network("100.64.0.0/10"):
            parser.error("--tailnet-ip must be a Tailscale IPv4 address")
        if args.tailnet_host and {args.port, args.model_port} & {TAILNET_HTTP_DASHBOARD, TAILNET_HTTP_MODEL}:
            parser.error("the two direct TCP ports must differ from the HTTPS Serve ports")
    roots = list(dict.fromkeys(p.expanduser().resolve() for p in [Path.home() / "models", *args.models_dir]))
    runner = Runner(find_llama(args.llama_server), args.model_port, roots)

    def shutdown(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    with Monitor(args.port, runner, args.tailnet_host, args.tailnet_ip, args.api_base_url) as monitor:
        print(f"Open http://{LOOPBACK}:{args.port}", flush=True)
        Thread(target=monitor.sample_forever, daemon=True).start()
        try:
            monitor.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stopped.set()
            runner.close()


if __name__ == "__main__":
    main()
