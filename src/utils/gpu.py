from typing import Tuple, Optional
import torch
from loguru import logger


def check_gpu_memory(device_id: int = 0) -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    try:
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        cached = torch.cuda.memory_reserved(device_id) / (1024**3)
        free = total - allocated
        return total, free, allocated
    except Exception as e:
        logger.error(f"GPU memory check failed: {e}")
        return 0.0, 0.0, 0.0


def get_optimal_device(min_memory_gb: float = 4.0) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        total, free, _ = check_gpu_memory()
        if free >= min_memory_gb:
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
