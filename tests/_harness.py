"""Shared scaffolding for the unit suites in this folder.

Deliberately not pytest. These suites run as plain scripts

    python tests/unit_adapters.py

so they work on a bare Colab runtime or inside the Docker image with no extra
dependency and no test-runner configuration, and they print the same PASS/FAIL table
the rest of this project's check scripts print.

Every suite here is CPU-only, needs no dataset and downloads nothing: models are built
from Settings and fed synthetic tensors of the right shape.
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeleton.snn_config import Settings  # noqa: E402

# cfg.FRAMEWORK selector -> the shim class main.py's FRAMEWORK_MODULES resolves to.
MODEL_CLASSES = {
    "torch": ("frameworks.snn_torch", "SNN_TORCH"),
    "norse": ("frameworks.snn_norse", "SNN_NORSE"),
    "sj": ("frameworks.snn_spikingjelly", "SNN_SJ"),
    "sinabs": ("frameworks.snn_sinabs", "SNN_SINABS"),
}
FRAMEWORKS = list(MODEL_CLASSES)
SLOTS = ["lif1", "lif2", "lif_out"]


class Suite:
    """Collects PASS/FAIL rows and prints them as one table."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[tuple[bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.rows.append((bool(condition), f"{name}{'  -- ' + detail if detail else ''}"))

    def expect_raises(self, name: str, exc_type, fn, must_mention: list[str] | None = None) -> None:
        """Assert fn() raises exc_type AND that the message names each must_mention token.

        The message is tested, not just the type: an error that fires without saying
        which key was wrong costs as much time as no error at all.
        """
        try:
            fn()
        except exc_type as error:
            missing = [t for t in (must_mention or []) if t not in str(error)]
            self.check(name, not missing,
                       f"message omits {missing}" if missing else f"raised {exc_type.__name__}")
        except Exception as error:  # noqa: BLE001 - wrong type is itself the failure
            self.check(name, False,
                       f"raised {type(error).__name__}, expected {exc_type.__name__}: {error}")
        else:
            self.check(name, False, "did not raise")

    def run(self, tests: list) -> int:
        for test in tests:
            try:
                test()
            except Exception as error:  # noqa: BLE001 - a crashing test is a failing test
                self.check(f"{test.__name__} (crashed)", False,
                           f"{type(error).__name__}: {error}")
        return self.report()

    def report(self) -> int:
        print("=" * 74)
        print(self.name)
        print("=" * 74)
        for ok, name in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        passed = sum(1 for ok, _ in self.rows if ok)
        print("-" * 74)
        print(f"{passed}/{len(self.rows)} passed")
        return 0 if passed == len(self.rows) else 1


def fresh_cfg() -> Settings:
    """A CPU Settings whose mutable blocks are deep-copied, so a test can edit
    NEURON / NEURON_TYPES without leaking into the next test or the real YAML."""
    cfg = Settings()
    cfg.NEURON = copy.deepcopy(cfg.NEURON)
    cfg.NEURON_TYPES = copy.deepcopy(cfg.NEURON_TYPES)
    cfg.NUM_CLASSES = 10
    cfg.DEVICE = "cpu"
    return cfg


def spike_input(cfg, time_steps: int = 8, batch: int = 3, seed: int = 7) -> torch.Tensor:
    """Binary input, like real event frames.

    Uniform noise leaves the readout silent, which would make a "does it fire" check
    pass vacuously -- so these are 0/1 values at roughly 50% density instead.
    """
    torch.manual_seed(seed)
    shape = (time_steps, batch, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
    return (torch.rand(*shape) < 0.5).float()


def build_model(framework: str, cfg=None):
    """The model exactly as learning/main.py builds it: through the shim class."""
    from skeleton.seeding import seed_model_init

    cfg = cfg or fresh_cfg()
    module_name, class_name = MODEL_CLASSES[framework]
    model_cls = getattr(importlib.import_module(module_name), class_name)
    seed_model_init(cfg.SEED)
    return model_cls(cfg), cfg


def single_neuron(framework: str, cfg=None):
    """One LIF layer on its own, for probing neuron dynamics without a network."""
    from frameworks.adapters import lif_factory

    cfg = cfg or fresh_cfg()
    return lif_factory(framework, cfg)("lif1")
