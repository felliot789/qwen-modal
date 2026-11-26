"""Sample Linux kernel counters without changing hardware settings."""

from collections import deque
from pathlib import Path
import os
import re
import shutil
import socket
import subprocess
import time


PROC = Path("/proc")
SYS = Path("/sys")
INTERVAL = 2
HISTORY = 120


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def number(path, scale=1):
    try:
        return int(read(path)) / scale
    except ValueError:
        return None


def fields(path, separator):
    return dict(line.split(separator, 1) for line in read(path).splitlines() if separator in line)


def temperatures(chip):
    sensors = {}
    for path in sorted(chip.glob("temp*_input")):
        label = read(path.with_name(path.name.replace("_input", "_label")))
        value = number(path, 1000)
        if value is not None:
            sensors[label or path.stem.split("_")[0]] = value
    return sensors


def gpu_name(device):
    if shutil.which("lspci") and re.fullmatch(r"[0-9a-f:.]+", device.resolve().name):
        try:
            result = subprocess.run(
                ["lspci", "-vmm", "-s", device.resolve().name],
                capture_output=True, text=True, timeout=2, check=True,
            )
            info = dict(line.split(":\t", 1) for line in result.stdout.splitlines() if ":\t" in line)
            return info.get("SDevice") or info.get("Device") or "Graphics adapter"
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
    return "Graphics adapter"


def llama_processes():
    processes = []
    for folder in PROC.iterdir():
        if not folder.name.isdecimal() or read(folder / "comm") != "llama-server":
            continue
        try:
            status = fields(folder / "status", ":")
            state = status.get("State", "").split()
            if not state or state[0] in ("Z", "X"):
                continue
            args = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
            model = None
            for arg, following in zip(args, args[1:]):
                if arg in ("-m", "--model"):
                    model = Path(following).name
                    break
                if arg.startswith("--model="):
                    model = Path(arg.split("=", 1)[1]).name
                    break
            rss = status.get("VmRSS", "").split()
            processes.append({
                "pid": int(folder.name), "model": model,
                "rss": int(rss[0]) * 1024 if rss else None,
            })
        except (OSError, ValueError):
            continue
    return processes


class Telemetry:
    def __init__(self):
        self.host = socket.gethostname()
        info = fields(PROC / "cpuinfo", ":")
        self.cpu_name = next(
            (v.strip() for k, v in info.items() if k.strip() == "model name"), "Processor",
        )
        self.previous_cpu = {}
        self.previous_energy = {}
        self.history = deque(maxlen=HISTORY // INTERVAL + 1)
        self.hwmon = [(read(chip / "name"), chip) for chip in (SYS / "class/hwmon").glob("hwmon*")]
        self.cpu_sensors = [p for name, p in self.hwmon if name in ("coretemp", "k10temp", "zenpower")]
        self.ram_sensors = [p for name, p in self.hwmon if name in ("jc42", "spd5118")]
        self.board_sensors = [p for name, p in self.hwmon if name.startswith(("nct", "it87"))]
        self.gpu = next(
            (p for p in sorted((SYS / "class/drm").glob("card*/device"))
             if (p / "gpu_busy_percent").exists()),
            None,
        )
        self.gpu_name = gpu_name(self.gpu) if self.gpu else "Graphics adapter"
        cap = SYS / "class/powercap"
        domains = sorted(cap.glob("*:*"))
        self.packages = [p for p in domains if read(p / "name").startswith("package-")]
        self.dram = [p for p in domains if read(p / "name").lower() == "dram"]

    def cpu_load(self):
        usage = {}
        for line in read(PROC / "stat").splitlines():
            if not line.startswith("cpu"):
                break
            name, *ticks = line.split()
            # Guest time is already included in user and nice.
            counts = list(map(int, ticks[:8]))
            total = sum(counts)
            idle = counts[3] + counts[4]
            previous = self.previous_cpu.get(name)
            self.previous_cpu[name] = (total, idle)
            if previous and total > previous[0]:
                idle_fraction = (idle - previous[1]) / (total - previous[0])
                usage[name] = max(0, min(100, 100 * (1 - idle_fraction)))
        per_thread = [[int(k[3:]), v] for k, v in usage.items() if k != "cpu"]
        return usage.get("cpu"), per_thread

    def power(self, domains, now):
        watts = []
        for domain in domains:
            energy = number(domain / "energy_uj")
            previous = self.previous_energy.pop(domain, None)
            if energy is None:
                continue
            self.previous_energy[domain] = (now, energy)
            if previous:
                delta = energy - previous[1]
                if delta < 0:
                    delta += number(domain / "max_energy_range_uj") or 0
                if delta >= 0 and now > previous[0]:
                    watts.append(delta / 1_000_000 / (now - previous[0]))
        return sum(watts) if watts and len(watts) == len(domains) else None

    def gpu_data(self):
        if self.gpu is None:
            return {"name": self.gpu_name}
        chip = next(iter(sorted((self.gpu / "hwmon").glob("hwmon*"))), None)
        metrics = {
            "name": self.gpu_name, "usage": number(self.gpu / "gpu_busy_percent"),
            "vram_used": number(self.gpu / "mem_info_vram_used"),
            "vram_total": number(self.gpu / "mem_info_vram_total"),
            "memory_load": number(self.gpu / "mem_busy_percent"),
        }
        if chip:
            temps = temperatures(chip)
            watts = number(chip / "power1_average", 1_000_000)
            metrics.update(
                temperature=temps.get("edge", temps.get("temp1")),
                hotspot=temps.get("junction"), memory_temp=temps.get("mem"),
                watts=watts if watts is not None else number(chip / "power1_input", 1_000_000),
                fan=number(chip / "fan1_input"), clock=number(chip / "freq1_input", 1_000_000),
            )
        return metrics

    def sample(self):
        now = time.monotonic()
        cpu_usage, threads = self.cpu_load()
        cpu_temps = [(label, value) for p in self.cpu_sensors for label, value in temperatures(p).items()]
        package = [v for k, v in cpu_temps if k.startswith(("Package", "Tdie", "Tctl"))]
        clocks = [
            float(line.split(":", 1)[1]) for line in read(PROC / "cpuinfo").splitlines()
            if line.startswith("cpu MHz")
        ]
        cpu = {
            "name": self.cpu_name, "usage": cpu_usage, "threads": threads,
            "logical_count": os.cpu_count() or 1,
            "load": list(os.getloadavg()), "clock": sum(clocks) / len(clocks) if clocks else None,
            "temperature": max(package or [v for _, v in cpu_temps], default=None),
            "core_temps": [[k, v] for k, v in cpu_temps if k.startswith("Core ")],
            "watts": self.power(self.packages, now),
            "power_restricted": any(not os.access(p / "energy_uj", os.R_OK) for p in self.packages),
        }
        mem = {k: int(v.split()[0]) * 1024 for k, v in fields(PROC / "meminfo", ":").items()}
        total = mem["MemTotal"]
        available = mem["MemAvailable"]
        ram = {
            "total": total, "available": available, "used": total - available,
            "usage": 100 * (total - available) / total,
            "swap_total": mem["SwapTotal"], "swap_used": mem["SwapTotal"] - mem["SwapFree"],
            "temperature": max((v for p in self.ram_sensors for v in temperatures(p).values()), default=None),
            "watts": self.power(self.dram, now),
            "power_restricted": any(not os.access(p / "energy_uj", os.R_OK) for p in self.dram),
        }
        gpu = self.gpu_data()
        fans = {}
        for chip in self.board_sensors:
            for path in sorted(chip.glob("fan*_input")):
                label = read(path.with_name(path.name.replace("_input", "_label")))
                rpm = number(path)
                if rpm:
                    fans[label or path.name.split("_")[0].replace("fan", "Header ")] = rpm
        self.history.append((now, cpu_usage, gpu.get("usage"), ram["usage"]))
        history = [[now - t, a, b, c] for t, a, b, c in self.history if now - t <= HISTORY]
        return {
            "host": self.host, "updated": time.time(), "cpu": cpu, "gpu": gpu, "ram": ram,
            "board_fans": fans, "history": history, "processes": llama_processes(),
        }
