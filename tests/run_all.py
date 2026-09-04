"""Run every unit suite in this folder and print one summary.

    python tests/run_all.py            all suites
    python tests/run_all.py adapters   only suites whose name contains "adapters"

Exit code is 0 only if every suite passes, so this works as a CI or pre-push gate.

Each suite runs in its own subprocess. That is deliberate: several of them mutate
global torch RNG state and build models from four different SNN libraries, and a
shared interpreter would let one suite's leftovers decide another suite's result --
which is the same class of bug (state leaking between frameworks in one process) that
this whole unit exists to remove from the pipeline itself.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Ordered cheapest-and-most-fundamental first, so the earliest failure is usually the
# most informative one: config access, then the neuron, then the picker, then the
# network, then the host pipeline's own functions.
SUITES = [
    "unit_neuron_spec.py",
    "unit_adapters.py",
    "unit_neuron_picker.py",
    "unit_shared_net.py",
    "unit_pipeline_integration.py",
    "unit_cli_config.py",
    "unit_seeding.py",
    "unit_training_metrics.py",
    "unit_capacity_metrics.py",
    "unit_results_schema.py",
    "unit_layer_naming.py",
    "unit_entrypoints.py",
]


def parse_tally(output: str) -> str:
    for line in reversed(output.strip().splitlines()):
        if line.strip().endswith("passed"):
            return line.strip()
    return "no tally reported"


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    suites = [s for s in SUITES if not wanted or any(w in s for w in wanted)]
    if not suites:
        print(f"no suite matches {wanted}. Available: {SUITES}")
        return 1

    rows: list[tuple[str, bool, str, float]] = []
    for suite in suites:
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(TESTS_DIR / suite)],
            capture_output=True, text=True, cwd=str(TESTS_DIR.parent),
        )
        elapsed = time.perf_counter() - started
        ok = completed.returncode == 0
        rows.append((suite, ok, parse_tally(completed.stdout), elapsed))

        if not ok:
            # Only a failing suite prints its detail, so a green run stays readable.
            print(f"\n{'=' * 74}\nFAILURES in {suite}\n{'=' * 74}")
            for line in completed.stdout.splitlines():
                if line.strip().startswith("FAIL"):
                    print(line)
            if completed.stderr.strip():
                print(completed.stderr.strip()[-2000:])

    print(f"\n{'=' * 74}")
    print(f"{'suite':<34}{'result':<10}{'tally':<18}{'seconds':>9}")
    print("-" * 74)
    for suite, ok, tally, elapsed in rows:
        print(f"{suite:<34}{'PASS' if ok else 'FAIL':<10}{tally:<18}{elapsed:>9.1f}")
    print("-" * 74)
    failed = [s for s, ok, _, _ in rows if not ok]
    print(f"{len(rows) - len(failed)}/{len(rows)} suites passed"
          + (f"  --  failing: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
