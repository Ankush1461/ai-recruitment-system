# ================================================================
# ZeroGPU - auto-detection, device pinning, CPU fallback
# ================================================================
# Hugging Face ZeroGPU Spaces make torch.cuda.is_available() return True even
# OUTSIDE @spaces.GPU functions (CUDA emulation mode), but any real CUDA
# operation raises there ("Low-level CUDA init reached...").
# This module:
#   1. auto-detects the ZeroGPU runtime (HF sets SPACES_ZERO_GPU=true) unless
#      an explicit ZEROGPU_ENABLED=0/1 overrides it (see config.py);
#   2. picks the device by PROBING for a real GPU - "cuda" only inside a
#      ZeroGPU worker process (real CUDA), "cpu" everywhere else (plain CPU
#      Spaces, local dev, and the ZeroGPU main process where CUDA is emulated);
#   3. wraps the local-model functions with @spaces.GPU when the runtime is
#      present, and falls back to plain CPU whenever a GPU call fails (daily
#      quota exhausted, queue rejection, emulation errors...) - latching off
#      for the rest of the process so we never queue-and-fail repeatedly.
#
# The `import spaces` here MUST stay above any torch/transformers import in
# the consumer modules (HF requires spaces before CUDA modules); this module
# imports no torch at module level.

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

import config

# Real `spaces` package - only provided by the ZeroGPU runtime (it is not a
# dependency of this project, so plain CPU Spaces / local dev leave it None
# and wrap_gpu() is a pass-through).
try:
    import spaces  # type: ignore
except Exception:  # pragma: no cover - CPU Spaces / local dev
    spaces = None  # type: ignore[assignment]

_healthy_lock = threading.Lock()
# Once a GPU call fails (quota exhausted, queue rejection, ...), retrying GPU
# would just queue and fail again - latch the process to CPU.
_gpu_healthy = True


def enabled() -> bool:
    """True when the ZeroGPU runtime is active (auto-detected or explicit)."""
    return config.ZEROGPU_ENABLED


def pick_device() -> str:
    """Return "cuda" only when a REAL GPU is reachable - i.e. inside a
    ZeroGPU worker process (forked with real CUDA after the runtime unpatches
    torch). Returns "cpu" on plain CPU Spaces, local dev, and the ZeroGPU
    MAIN process, where CUDA is emulated and the probe raises.

    Probing every call is intentional: the worker is a fork of the main
    process, so a cached "cpu" result (or a cached "cuda" one) would leak
    across the fork. The probe is cheap (a microsecond-scale op or an
    immediate raise).
    """
    if not enabled():
        return "cpu"
    try:
        import torch

        torch.zeros(1).cuda()
        return "cuda"
    except Exception:
        return "cpu"


def wrap_gpu(fn: Callable) -> Callable:
    """Route fn through @spaces.GPU when running on ZeroGPU, with automatic
    CPU fallback: any GPU failure (quota exceeded, queue rejection, ...)
    re-runs fn on plain CPU and latches the process to CPU.

    No-op on non-ZeroGPU hosts - the decorator is effect-free there anyway.
    """
    if not enabled() or spaces is None:
        return fn
    gpu_fn = spaces.GPU(fn)

    @functools.wraps(fn)
    def _run(*args: Any, **kwargs: Any) -> Any:
        global _gpu_healthy
        if not _gpu_healthy:
            return fn(*args, **kwargs)
        try:
            return gpu_fn(*args, **kwargs)
        except Exception as exc:  # any GPU failure -> CPU retry
            with _healthy_lock:
                was = _gpu_healthy
                _gpu_healthy = False
            if was:
                print(
                    f"[zerogpu] GPU call failed ({type(exc).__name__}: {exc}) - "
                    "falling back to CPU for this process.",
                    flush=True,
                )
            return fn(*args, **kwargs)

    return _run
