import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

import config


def set_seed(seed: int | None = None):
    seed = config.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    def __init__(self, device: torch.device | None = None):
        self.device = device

    def __enter__(self):
        if self.device is not None:
            sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None:
            sync(self.device)
        self.seconds = time.perf_counter() - self.start


def results_dir(smoke: bool) -> Path:
    name = ("results_smoke" if smoke else "results") + (f"_{config.TAG}" if config.TAG else "")
    d = config.ROOT / name
    for sub in ("", "checkpoints", "runs", "figures", "downstream"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hardware_info(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info
