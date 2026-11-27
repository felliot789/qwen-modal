"""Publish averaged RAPL CPU and DRAM power for the unprivileged dashboard."""

import json
from pathlib import Path
import time


RUNTIME = Path("/run/local-llm-rapl")
POWERCAP = Path("/sys/class/powercap")


def sample(domains):
    return [int((domain / "energy_uj").read_text()) for domain in domains]


def main():
    domains = {"cpu": [], "dram": []}
    for path in sorted(POWERCAP.glob("intel-rapl:*")):
        name = (path / "name").read_text().strip()
        if name.startswith("package-"):
            domains["cpu"].append(path)
        elif name.lower() == "dram":
            domains["dram"].append(path)

    ranges = {
        key: [int((path / "max_energy_range_uj").read_text()) for path in paths]
        for key, paths in domains.items()
    }
    previous = {key: sample(paths) for key, paths in domains.items()}
    last = time.monotonic()
    temporary = RUNTIME / "power.tmp"
    while True:
        time.sleep(2)
        current = {key: sample(paths) for key, paths in domains.items()}
        now = time.monotonic()
        watts = {
            key: round(sum((end - start) % limit for start, end, limit in
                           zip(previous[key], values, ranges[key])) / (now - last) / 1e6, 2)
            for key, values in current.items() if values
        }
        temporary.write_text(json.dumps({"monotonic": now, **watts}))
        temporary.replace(RUNTIME / "power.json")
        previous, last = current, now


if __name__ == "__main__":
    main()
