from __future__ import annotations

from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def gpu_status() -> dict[str, Any]:
    torch = _torch()
    if torch is None:
        return {
            "available": False,
            "reason": "PyTorch is not installed",
            "torchVersion": None,
            "name": None,
            "totalMemoryBytes": None,
            "freeMemoryBytes": None,
        }
    available = bool(torch.cuda.is_available())
    status: dict[str, Any] = {
        "available": available,
        "reason": None if available else "CUDA is unavailable",
        "torchVersion": torch.__version__,
        "name": None,
        "totalMemoryBytes": None,
        "freeMemoryBytes": None,
    }
    if available:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        status.update(
            name=properties.name,
            totalMemoryBytes=total_bytes,
            freeMemoryBytes=free_bytes,
        )
    return status


def require_cuda() -> str:
    status = gpu_status()
    if not status["available"]:
        raise RuntimeError(status["reason"] or "CUDA is unavailable")
    return "cuda:0"

