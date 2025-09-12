import os
import time
from pathlib import Path

def set_cpu_threads(threads: int | None):
    if threads is None or threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass

def human_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    for unit in ["KB","MB","GB","TB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def dir_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def average_inference_time(runtimes: list[float]) -> float:
    if not runtimes: return float("nan")
    return sum(runtimes) / len(runtimes)

def measure_time(func, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    func(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0
