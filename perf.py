# =============================================================================
# utils/perf.py — lightweight cost instrumentation (wall-clock, peak RSS,
# device-synchronized timing). Used by the BGL validation so every reported
# "training time / inference time / memory" number is MEASURED, not estimated.
# No new dependencies: stdlib resource/time + torch for device sync.
# =============================================================================

import resource
import sys
import time

import torch


class Timer:
    """Wall-clock context manager: `with Timer() as t: ... ; t.seconds`."""

    def __enter__(self):
        self.seconds = 0.0
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


def peak_rss_mb() -> float:
    """Peak resident set size of THIS process in MiB (cumulative since start).

    ru_maxrss is reported in bytes on macOS but kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return peak / divisor


def device_sync(device) -> None:
    """Block until queued device work finishes, so surrounding timings are honest."""
    if device is None:
        return
    dtype = device.type if hasattr(device, "type") else str(device)
    if dtype == "mps":
        torch.mps.synchronize()
    elif dtype == "cuda":
        torch.cuda.synchronize()
