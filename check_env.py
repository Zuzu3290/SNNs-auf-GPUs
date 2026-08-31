"""Print the versions of everything this project depends on, and check them.

Run this FIRST on any new machine -- your laptop, Colab, Kaggle -- to confirm the
environment is complete and matches requirements.txt before spending GPU time. Paste
its output into the thesis appendix: framework benchmarks are version-sensitive, so
these numbers are a result too.

    python check_env.py
    python check_env.py --strict     # exit non-zero on a version MISMATCH too

Ported from the SNNs_2 comparison pipeline. Exits non-zero when a package is missing,
so it works as a first cell in a notebook or a step in a script.

WHY A VERSION MISMATCH MATTERS HERE. This project compares SNN frameworks. If Colab
resolves snntorch to a different version than the laptop did, the comparison is partly
between versions rather than between frameworks -- and nothing in the numbers says so.
requirements.txt pins the four framework libraries, tonic and numpy exactly for that
reason; this script is what proves the pins took.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# (import name, pip name). They differ for several of these -- pynvml ships as
# nvidia-ml-py, yaml as PyYAML.
PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    # the four under comparison
    ("snntorch", "snntorch"),
    ("norse", "norse"),
    ("spikingjelly", "spikingjelly"),
    ("sinabs", "sinabs"),
    # sinabs pulls nir and nirtorch and imports them itself; nothing here uses them
    # directly, but their versions belong in the appendix.
    ("nir", "nir"),
    ("nirtorch", "nirtorch"),
    # event data
    ("tonic", "tonic"),
    ("numpy", "numpy"),
    ("numba", "numba"),
    ("h5py", "h5py"),
    # metrics, monitoring, output
    ("pynvml", "nvidia-ml-py"),
    ("psutil", "psutil"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
]

# Pinned in requirements.txt because they decide the RESULTS. A mismatch on any of
# these makes a cross-machine comparison partly a comparison of versions.
RESULT_CRITICAL = {"snntorch", "norse", "spikingjelly", "sinabs", "tonic", "numpy", "torch"}

# Reported from pip metadata only -- NEVER imported.
#
# samna is SynSense's chip SDK and an optional sinabs extra. Importing it makes sinabs
# try to pip-install it from a private GitLab index, so an `import samna` here would
# turn a read-only environment check into a network install that can fail and take the
# check down with it. (Verified: sinabs 3.1.3 imports perfectly well without samna --
# it is only needed to talk to real Speck hardware, which this pipeline never does.)
METADATA_ONLY: list[tuple[str, str]] = [("samna", "samna")]


def package_versions() -> dict[str, str]:
    """Version string per package, or an explicit failure marker. Never raises."""
    found: dict[str, str] = {}
    for import_name, pip_name in PACKAGES:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001 - any failure is worth reporting
            found[pip_name] = f"NOT INSTALLED ({type(exc).__name__})"
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            # spikingjelly exposes no __version__; fall back to pip metadata.
            try:
                from importlib.metadata import version as pkg_version

                version = pkg_version(pip_name)
            except Exception:  # noqa: BLE001
                version = "installed (version unknown)"
        found[pip_name] = str(version)

    # Metadata only, no import -- see METADATA_ONLY for why. Absence is reported as
    # "not installed (optional)" rather than a failure: nothing here needs it.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    for _import_name, pip_name in METADATA_ONLY:
        try:
            found[pip_name] = f"{pkg_version(pip_name)}  (metadata only, not imported)"
        except PackageNotFoundError:
            found[pip_name] = "not installed (optional -- Speck hardware only)"
    return found


def required_pins(path: Path) -> dict[str, str]:
    """The `name==version` lines from requirements.txt. Comments and unpinned lines
    are skipped -- only exact pins can be checked."""
    pins: dict[str, str] = {}
    if not path.is_file():
        return pins
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def compare(installed: dict[str, str], pins: dict[str, str]) -> list[tuple[str, str, str, bool]]:
    """(package, wanted, got, critical) for every pin that does not match.

    torch's local build tag is ignored: 2.13.0+cpu and 2.13.0+cu128 are the same
    version built for different hardware, and requirements.txt pins the version
    deliberately so the file works on both.
    """
    problems = []
    for name, wanted in sorted(pins.items()):
        got = installed.get(name)
        if got is None or got.startswith("NOT INSTALLED"):
            continue  # already reported as missing
        if got.split("+")[0] != wanted.split("+")[0]:
            problems.append((name, wanted, got, name in RESULT_CRITICAL))
    return problems


def gpu_info() -> dict[str, str]:
    """GPU name, memory and driver. Empty dict if torch is missing entirely."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {}

    # NVML is probed regardless of whether TORCH can use the GPU: a CPU-only torch
    # build on a machine with an NVIDIA card still reports power perfectly well, and
    # that is worth knowing before assuming energy cannot be measured here.
    if not torch.cuda.is_available():
        info = {"cuda": "not available to torch (CPU-only build or no GPU)"}
    else:
        info = {
            "cuda": torch.version.cuda or "unknown",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": str(torch.cuda.device_count()),
            "gpu_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}",
        }

    # Driver and power-reading support both come from NVML, which is what the energy
    # metric uses. If the power read fails here, energy cannot be measured on this
    # machine and the energy columns will be empty.
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["nvml_driver"] = pynvml.nvmlSystemGetDriverVersion()
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
            info["nvml_power_readable"] = f"yes ({milliwatts / 1000:.1f} W right now)"
        except Exception as exc:  # noqa: BLE001
            info["nvml_power_readable"] = f"NO -- energy metric unavailable ({exc})"
        pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        info["nvml_driver"] = f"NVML unavailable ({exc})"

    return info


def cpu_info() -> dict[str, str]:
    """Physical cores and free RAM.

    Reported because this pipeline is frequently DATA-bound rather than GPU-bound: a
    Colab runtime with 1 physical core has been measured holding GPU utilisation near
    11%, with the loader unable to keep up. A framework comparison run under that
    condition measures the data pipeline, not the frameworks.
    """
    try:
        import psutil
    except Exception as exc:  # noqa: BLE001
        return {"cpu": f"psutil unavailable ({exc})"}

    physical = psutil.cpu_count(logical=False) or 0
    memory = psutil.virtual_memory()
    info = {
        "cpu_cores_physical": str(physical),
        "cpu_cores_logical": str(psutil.cpu_count(logical=True) or 0),
        "ram_total_gb": f"{memory.total / 1024**3:.1f}",
        "ram_available_gb": f"{memory.available / 1024**3:.1f}",
    }
    if physical and physical <= 2:
        info["warning"] = (
            f"only {physical} physical core(s) -- the DataLoader will likely starve the "
            "GPU, and timing numbers will reflect data prep rather than the framework"
        )
    return info


def norse_build() -> dict[str, str]:
    """Is Norse running compiled C++ here, or pure Python?

    v1.1.0's release notes say it "transformed Norse into a Python-only module by
    eliminating C++ code", so pure Python is EXPECTED rather than a failed build.
    Recorded anyway, because it belongs in the write-up: this Norse is a pure-Python
    implementation being compared against libraries that may use fused kernels.
    """
    try:
        import norse
    except Exception as exc:  # noqa: BLE001
        return {"norse_build": f"not importable ({type(exc).__name__})"}

    root = Path(norse.__file__).parent
    compiled = [p.name for p in root.rglob("*.so")] + [p.name for p in root.rglob("*.pyd")]
    return {
        "norse_version": getattr(norse, "__version__", "unknown"),
        "norse_build": (
            f"compiled extensions present: {compiled}" if compiled
            else "pure Python (no compiled extension) -- expected for v1.1.0"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Module level so tests can assert the advertised flag set matches HOW_TO_RUN.md."""
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="also exit non-zero when an installed version differs from "
                             "the pin in requirements.txt")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    print("=" * 68)
    print("ENVIRONMENT")
    print("=" * 68)
    print(f"{'python':<24} {platform.python_version()}")
    print(f"{'platform':<24} {platform.system()} {platform.release()}")
    print()

    versions = package_versions()
    for name, version in versions.items():
        print(f"{name:<24} {version}")

    print()
    print("=" * 68)
    print("CPU / RAM")
    print("=" * 68)
    for key, value in cpu_info().items():
        print(f"{key:<24} {value}")

    print()
    print("=" * 68)
    print("GPU")
    print("=" * 68)
    info = gpu_info()
    if not info:
        print("torch missing -- cannot query GPU")
    for key, value in info.items():
        print(f"{key:<24} {value}")

    print()
    print("=" * 68)
    print("NORSE BUILD")
    print("=" * 68)
    for key, value in norse_build().items():
        print(f"{key:<24} {value}")

    # ---- verdict ------------------------------------------------------------------
    missing = [n for n, v in versions.items() if v.startswith("NOT INSTALLED")]
    mismatches = compare(versions, required_pins(REPO_ROOT / "requirements.txt"))

    print()
    print("=" * 68)
    print("VERDICT")
    print("=" * 68)

    if mismatches:
        print("version mismatches against requirements.txt:")
        for name, wanted, got, critical in mismatches:
            mark = "  RESULT-CRITICAL" if critical else ""
            print(f"  {name:<20} wanted {wanted:<12} got {got}{mark}")
        if any(critical for *_, critical in mismatches):
            print()
            print("  A result-critical mismatch means a comparison run on THIS machine is")
            print("  not directly comparable with one run elsewhere. Fix the version, or")
            print("  record the difference alongside the results.")
        print()

    if missing:
        print(f"FAIL: missing packages -> {', '.join(missing)}")
        return 1
    if mismatches and args.strict:
        print("FAIL (--strict): versions differ from requirements.txt")
        return 1
    if mismatches:
        print("OK: everything imports, but versions differ from the pins above.")
        return 0
    print("OK: all packages import, and every pinned version matches requirements.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
