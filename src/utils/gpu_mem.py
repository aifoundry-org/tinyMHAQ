import functools
import shutil
import subprocess

# 1) NVIDIA NVML fallback
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    _USE_NVML = True
except ImportError:
    _USE_NVML = False

# 2) AMD ROCm SMI fallback
_USE_ROCMSMI = shutil.which("rocm-smi") is not None

def _get_gpu_memory(gpu_index: int = 0):
    """
    Return (used_MB, total_MB) for GPU at gpu_index.
    Tries NVML first, then rocm-smi.
    """
    if _USE_NVML:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        info = nvmlDeviceGetMemoryInfo(handle)
        used  = int(info.used  // 1024**2)
        total = int(info.total // 1024**2)
        return used, total

    elif _USE_ROCMSMI:
        # CSV output: header + one line per card: device,TotalBytes,UsedBytes
        cmd = ["rocm-smi", "--showmeminfo", "vram", "--csv"]
        out = subprocess.check_output(cmd, encoding="utf-8")
        lines = [l for l in out.strip().splitlines() if l and not l.startswith("#")]
        if len(lines) < 2 or gpu_index >= len(lines) - 1:
            raise IndexError(f"GPU index {gpu_index} out of range")
        # skip header
        parts = lines[gpu_index + 1].split(",")
        total_bytes = int(parts[1])
        used_bytes  = int(parts[2])
        used  = used_bytes  // 1024**2
        total = total_bytes // 1024**2
        return used, total

    else:
        raise RuntimeError(
            "No supported GPU monitoring backend found. "
            "Install pynvml (for NVIDIA) or ensure rocm-smi is on your PATH (for AMD)."
        )

def measure_gpu_mem(gpu_index: int = 0):
    """
    Decorator factory: wraps a function and prints GPU memory usage
    before & after its execution (for the specified GPU index).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            before_used, before_total = _get_gpu_memory(gpu_index)
            result = fn(*args, **kwargs)
            after_used, after_total = _get_gpu_memory(gpu_index)
            delta = after_used - before_used
            print(
                f"[GPU{gpu_index}] Memory used: "
                f"{before_used} → {after_used} MiB  (Δ {delta:+d} MiB)"
            )
            return result
        return wrapper
    return decorator
