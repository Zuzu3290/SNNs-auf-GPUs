# Python wrapper over the C++ GPU preflight diagnostic (skeleton/gpu_diagnostics.cu).
# Calls snn_runtime.run_gpu_preflight(), feeds results into ReliabilityTracker,
# and optionally blocks training if critical hardware issues are detected.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skeleton.reliability import ReliabilityTracker


def run_preflight(
    device_idx:       int                    = 0,
    reliability:      "ReliabilityTracker | None" = None,
    block_on_failure: bool                   = True,
) -> bool:
    """
    Run the C++ GPU hardware health check before training begins.

    Parameters
    ----------
    device_idx       : CUDA device index to check.
    reliability      : Optional ReliabilityTracker — anomalies are recorded as
                       interruptions or failures automatically.
    block_on_failure : If True (default), raises RuntimeError when the device
                       is unhealthy. Set False to log and continue.

    Returns True if the device is healthy, False otherwise (only when block_on_failure=False).
    """
    try:
        import snn_runtime as rt  # type: ignore[import]
    except ImportError:
        print("[GPU Preflight] snn_runtime not built — skipping hardware checks.")
        return True

    report = rt.run_gpu_preflight(device_idx)
    rt.print_diagnostic_report(report)

    if reliability is not None:
        if report.cuda_error_cleared:
            reliability.record_interruption(
                "Stale CUDA error cleared at startup — check previous session logs"
            )
        if report.nvml_available and report.ecc_corrected > 0:
            reliability.record_interruption(
                f"ECC correctable errors: {report.ecc_corrected} — monitor for increase"
            )
        if report.nvml_available and report.temperature_c >= 80:
            reliability.record_interruption(
                f"GPU temperature {report.temperature_c:.0f} C at startup — thermal headroom low"
            )
        if not report.healthy:
            reliability.record_failure(
                f"GPU preflight failed: {report.failure_reason}"
            )

    if not report.healthy:
        msg = (
            f"[GPU Preflight] Device {device_idx} failed health check: "
            f"{report.failure_reason}"
        )
        if block_on_failure:
            raise RuntimeError(msg)
        print(msg)
        return False

    return True
