import hashlib
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

app = modal.App("qwen38-27b")
volume = modal.Volume.from_name("qwen38-27b-model", create_if_missing=False)
VARIANT = os.environ.get("QWEN_VARIANT", "q8")
VARIANTS = {
    "q8": ("Q8_K_P", "L40S", "f16", "qwen38-hauhau-q8kp,qwen38-hauhau-iq4xs", "4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2"),
    "iq4": ("IQ4_XS", "L4", "q4_0", "qwen38-hauhau-iq4xs", "034b4c6b7c5cd9f4a2f99570d5097698f9d1e265a0fb000207223cbf9f9bd0b0"),
}
if VARIANT not in VARIANTS:
    raise ValueError("QWEN_VARIANT must be q8 or iq4")
QUANT, GPU, KV, ALIASES, SHA256 = VARIANTS[VARIANT]
MODEL = f"Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-{QUANT}.gguf"
REVISION = "993a5971fda8f30dd1b7eb2654792ba4415c7460"
URL = f"https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/{REVISION}/{MODEL}"
PATH = f"/models/{MODEL}"
ENV = {"QWEN_VARIANT": VARIANT}


@app.function(image=modal.Image.debian_slim(python_version="3.12"), env=ENV, volumes={"/models": volume}, cpu=1, memory=1024, timeout=1800)
def download():
    target = Path(PATH)
    if target.exists():
        raise RuntimeError("Model already exists; refusing to overwrite it")
    tmp = target.with_suffix(".download")
    digest = hashlib.sha256()
    with urllib.request.urlopen(URL, timeout=120) as response, tmp.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Model checksum mismatch")
    tmp.rename(target)
    volume.commit()
    print(f"Verified {target.name}: {target.stat().st_size} bytes")


# Exact CUDA image used by the successful Q8 127K-token benchmark.
image = modal.Image.from_id("im-7D6gCUOQJQr4KuMaNYpmSz")


@app.server(image=image, env=ENV, gpu=GPU, cpu=(1, 2), memory=2048, volumes={"/models": volume.read_only()}, port=8000,
            target_concurrency=1, min_containers=0, max_containers=1, scaledown_window=30,
            startup_timeout=180, exit_grace_period=10, routing_region="us-east", unauthenticated=False)
class Server:
    @modal.enter()
    def start(self):
        if not Path(PATH).is_file():
            raise RuntimeError(f"Model is missing: {MODEL}")
        cmd = [
            "/app/llama-server", "-m", PATH, "--host", "0.0.0.0", "--port", "8000",
            "--alias", ALIASES, "-c", "131072", "-np", "1", "-ngl", "all",
            "-fa", "on", "-ctk", KV, "-ctv", KV, "-b", "512", "-ub", "256", "-t", "2", "-tb", "2",
            "-fit", "off", "--jinja", "--perf", "--no-webui",
        ]
        if VARIANT == "q8":
            cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0"]
        self.proc = subprocess.Popen(cmd)
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited: {self.proc.returncode}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], check=False)
                            return
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            raise TimeoutError("llama-server did not become ready")
        except BaseException:
            self.stop()
            raise

    @modal.exit()
    def stop(self):
        if not hasattr(self, "proc") or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
