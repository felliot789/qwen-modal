import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

app = modal.App("qwen38-q8-bench")
volume = modal.Volume.from_name("qwen38-27b-model", create_if_missing=False)
name = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
path = Path("/models") / name
sha256 = "4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2"
url = "https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/993a5971fda8f30dd1b7eb2654792ba4415c7460/" + name
image = modal.Image.from_id("im-7D6gCUOQJQr4KuMaNYpmSz")
mtp = os.environ.get("QWEN_BENCH_MTP", "0")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/models": volume}, cpu=1, memory=1024, timeout=1800)
def download():
    if path.exists():
        raise RuntimeError("Q8 file exists; refusing to overwrite it")
    temp = path.with_suffix(".download")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        temp.unlink(missing_ok=True)
        raise RuntimeError("Q8 checksum mismatch")
    temp.rename(path)
    volume.commit()
    print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)


def memory():
    root = Path("/sys/fs/cgroup")
    result = {}
    for filename in ("memory.current", "memory.peak", "memory.stat"):
        try:
            result[filename] = (root / filename).read_text()
        except OSError:
            pass
    return result


def gpu():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True).strip()


@app.function(image=image, env={"QWEN_BENCH_MTP": mtp}, gpu="L40S", cpu=(1, 2), memory=2048, volumes={"/models": volume.read_only()}, timeout=1200)
def benchmark():
    if not path.is_file():
        raise RuntimeError("Verified model file missing")
    cmd = ["/app/llama-server", "-m", str(path), "--host", "127.0.0.1", "--port", "8000",
           "--alias", "qwen38-q8", "-c", "131072", "-np", "1", "-ngl", "all", "-fa", "on",
           "-ctk", "f16", "-ctv", "f16", "-b", "512", "-ub", "256", "-t", "2", "-tb", "2",
           "-fit", "off", "--jinja", "--perf", "--no-webui", "-lv", "4"]
    if mtp == "1":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0"]
    log_path = Path("/tmp/qwen-q8-server.log")
    with log_path.open("w+") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while time.monotonic() - start < 300:
                if proc.poll() is not None:
                    log.flush()
                    raise RuntimeError("Server exited: " + log_path.read_text(errors="replace")[-8000:])
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError("Server did not become ready")
            print("LOADED", json.dumps({"seconds": round(time.monotonic() - start, 2), "gpu": gpu(), "memory": memory()}), flush=True)

            def complete(prompt, tokens):
                body = {"model": "qwen38-q8", "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
                request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=850) as response:
                    answer = json.load(response)
                return {"answer": answer["choices"][0]["message"].get("content"), "usage": answer.get("usage"),
                        "timings": answer.get("timings"), "gpu": gpu(), "memory": memory()}

            print("SHORT", json.dumps(complete("In two sentences, explain why GPU memory bandwidth matters for language models.", 100)), flush=True)
            prompt = "Remember this access code: PLUM-4827. Here is filler text: " + " apple" * 127000 + "\nWhat was the access code? Reply with the code only."
            print("LONG", json.dumps(complete(prompt, 24)), flush=True)
            log.flush()
            for line in log_path.read_text(errors="replace").splitlines():
                if any(x in line.lower() for x in ("offloaded", "kv buffer size", "cuda0 model buffer", "n_ctx_slot", "draft acceptance", "draft tokens")):
                    print("SERVER", line, flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
