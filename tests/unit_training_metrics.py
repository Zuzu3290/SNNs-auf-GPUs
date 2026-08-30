"""Unit tests for the training-loop corrections: warm-up, the firing window, and the
iteration cap.

    python tests/unit_training_metrics.py

Three fixes are covered, and each one was a number that came out wrong rather than a
crash -- which is why they need tests rather than a glance:

  D7   no warm-up, so CUDA kernel compilation was charged to training, and in a
       multi-framework run charged only to whichever framework ran first
  D1   the Hz window read a config attribute that does not exist, so every firing-rate
       figure was computed against a fixed 15 ms instead of the real sample duration
  D12  the iteration cap was checked after the batch had already been trained on, so an
       epoch ran num_iters + 1 iterations

CPU-only, no dataset, no download.
"""
from __future__ import annotations

import pathlib

import torch

from _harness import FRAMEWORKS, Suite, build_model, fresh_cfg, spike_input
from learning.utilities import (
    firing_window_seconds, spikes_per_neuron_per_inference, warm_up,
)
from skeleton.config_loader import deep_merge, load_base
from skeleton.snn_config import Settings
from skeleton.workflow_config import WorkflowSettings

import tempfile

suite = Suite("unit_training_metrics")
REPO_TMP = pathlib.Path(tempfile.mkdtemp(prefix="snn_unit_"))


def settings_pair(overrides: dict):
    merged = deep_merge(load_base(), overrides)
    return Settings(config=merged), WorkflowSettings(config=merged)


# ---------------------------------------------------------------------------
# 1. D1 -- the firing window
# ---------------------------------------------------------------------------
def test_window_is_none_when_not_knowable() -> None:
    """n_time_bins divides a recording of unknown length. Withholding Hz is the point:
    the previous code invented 15 ms and published figures ~20x too high."""
    cfg, wf = settings_pair({"binning": {"mode": "n_time_bins", "sample_duration_us": None}})
    suite.check("n_time_bins with no stated duration gives None",
                firing_window_seconds(cfg, wf) is None, str(firing_window_seconds(cfg, wf)))


def test_window_from_a_stated_sample_duration() -> None:
    cfg, wf = settings_pair({"binning": {"sample_duration_us": 300000}})
    suite.check("stated duration is used", firing_window_seconds(cfg, wf) == 0.3,
                str(firing_window_seconds(cfg, wf)))


def test_window_derived_from_time_window_framing() -> None:
    """time_window binning fixes each frame's duration, so T frames span T x that."""
    cfg, wf = settings_pair({"binning": {"mode": "time_window", "time_window_ms": 15.0,
                                         "n_time_bins": 16, "sample_duration_us": None}})
    suite.check("time_window binning derives T x window",
                abs(firing_window_seconds(cfg, wf) - 0.24) < 1e-12,
                str(firing_window_seconds(cfg, wf)))


def test_window_from_temporal_slicing_by_time() -> None:
    cfg, wf = settings_pair({"temporal": {"enabled": True, "slice_duration_us": 15000},
                             "binning": {"sample_duration_us": None}})
    suite.check("slicing by time uses the slice duration",
                firing_window_seconds(cfg, wf) == 0.015, str(firing_window_seconds(cfg, wf)))


def test_stated_duration_wins_over_derivation() -> None:
    cfg, wf = settings_pair({"binning": {"mode": "time_window", "time_window_ms": 15.0,
                                         "n_time_bins": 16, "sample_duration_us": 500000}})
    suite.check("an explicitly stated duration takes precedence",
                firing_window_seconds(cfg, wf) == 0.5, str(firing_window_seconds(cfg, wf)))


def test_the_attribute_the_old_code_read_does_not_exist() -> None:
    """The root cause, asserted so it cannot quietly come back."""
    cfg, wf = settings_pair({})
    suite.check("cfg has no TEMPORAL_SLICE_DURATION_US",
                not hasattr(cfg, "TEMPORAL_SLICE_DURATION_US"))
    suite.check("wf has no TEMPORAL_SLICE_DURATION_US",
                not hasattr(wf, "TEMPORAL_SLICE_DURATION_US"))
    suite.check("cfg has no TIMESTEPS", not hasattr(cfg, "TIMESTEPS"))
    # The slice duration now lives with the rest of the slicing config, on
    # WorkflowSettings, not on Settings.
    suite.check("the slice duration lives on wf", hasattr(wf, "SLICE_DURATION_US"))
    suite.check("and no longer on cfg", not hasattr(cfg, "TEMPORAL_SLICE_DURATION"))


def test_spikes_per_inference_is_time_unit_free() -> None:
    suite.check("rate x T", spikes_per_neuron_per_inference(0.05, 16) == 0.8)
    suite.check("a silent layer gives 0", spikes_per_neuron_per_inference(0.0, 16) == 0.0)
    suite.check("scales with T", spikes_per_neuron_per_inference(0.05, 32) == 1.6)


def test_the_size_of_the_old_error() -> None:
    """0.05 rate at T=16 is 0.8 spikes/neuron/inference. Over a real 300 ms N-MNIST
    sample that is 2.7 Hz; the old fixed 15 ms window reported 53.3 Hz."""
    per_inference = spikes_per_neuron_per_inference(0.05, 16)
    old_hz, true_hz = per_inference / 0.015, per_inference / 0.300
    suite.check("the old window overstates by exactly 20x",
                abs(old_hz / true_hz - 20.0) < 1e-9, f"{old_hz / true_hz:.4f}")
    suite.check("the error is a constant factor, so relative comparisons survived",
                abs((2 * per_inference / 0.015) / (2 * per_inference / 0.300) - 20.0) < 1e-9)


# ---------------------------------------------------------------------------
# 2. D7 -- warm-up
# ---------------------------------------------------------------------------
def test_warmup_leaves_the_weights_untouched() -> None:
    """The whole point: it compiles kernels without training. If the weights moved,
    the timed epochs would not start from the seeded state."""
    from skeleton.seeding import shared_weight_fingerprint

    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        before = shared_weight_fingerprint(model)
        report = warm_up(model, spike_input(cfg, 6, 2), 3)
        suite.check(f"warm-up reports weights unchanged: {framework}",
                    report["weights_unchanged"])
        suite.check(f"the fingerprint really is unchanged: {framework}",
                    shared_weight_fingerprint(model) == before)


def test_warmup_clears_gradients() -> None:
    """It runs backward, so gradients exist. Left behind, the first real step would
    apply warm-up gradients on top of the first batch's."""
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        warm_up(model, spike_input(cfg, 6, 2), 2)
        leftover = [n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None]
        suite.check(f"no gradients survive warm-up: {framework}", not leftover,
                    f"{len(leftover)} left")


def test_warmup_leaves_no_activity_recordings() -> None:
    """Warm-up spikes must not reach the metrics."""
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        warm_up(model, spike_input(cfg, 6, 2), 2)
        suite.check(f"activity buffers are empty after warm-up: {framework}",
                    all(v is None for v in model.activity.recordings().values()))


def test_warmup_zero_is_a_no_op() -> None:
    model, cfg = build_model("torch")
    report = warm_up(model, spike_input(cfg, 6, 2), 0)
    suite.check("0 iterations does nothing", report["iterations"] == 0)
    suite.check("and reports no change", report["weights_unchanged"] is True)


def test_warmup_actually_runs_the_requested_count() -> None:
    model, cfg = build_model("norse")
    suite.check("the reported count matches what was asked",
                warm_up(model, spike_input(cfg, 6, 2), 4)["iterations"] == 4)


def test_warmup_iterations_is_configurable() -> None:
    cfg = Settings()
    suite.check("warmup_iterations is in the shipped config",
                "warmup_iterations" in load_base()["training"])
    suite.check("Settings exposes it", hasattr(cfg, "WARMUP_ITERATIONS"))
    suite.check("it defaults to a non-zero count", cfg.WARMUP_ITERATIONS > 0,
                str(cfg.WARMUP_ITERATIONS))
    cfg2, _ = settings_pair({"training": {"warmup_iterations": 0}})
    suite.check("it can be switched off", cfg2.WARMUP_ITERATIONS == 0)


# ---------------------------------------------------------------------------
# 3. D12 -- the iteration cap
# ---------------------------------------------------------------------------
def cap_iterations(n_batches: int, num_iters: int) -> int:
    """The loop shape now used in SNNTrainer.train(): checked BEFORE the work."""
    ran = 0
    for i in range(n_batches):
        if i >= num_iters:
            break
        ran += 1
    return ran


def test_iteration_cap_runs_exactly_what_was_asked() -> None:
    for n_batches, num_iters in [(400, 3), (400, 10), (400, 100), (50, 49)]:
        expected = min(n_batches, num_iters)
        suite.check(f"loader {n_batches}, asked {num_iters} -> runs {expected}",
                    cap_iterations(n_batches, num_iters) == expected,
                    str(cap_iterations(n_batches, num_iters)))


def test_cap_above_the_loader_length_is_harmless() -> None:
    suite.check("a cap larger than the loader just exhausts it",
                cap_iterations(5, 400) == 5)


def test_the_calibrated_case_is_unaffected() -> None:
    """Why the bug was invisible in normal use: with calibrate_batch_size on,
    ITERA == len(loader), so the loader runs out before the cap is ever reached."""
    suite.check("ITERA == loader length runs the whole loader",
                cap_iterations(400, 400) == 400)


def test_the_old_shape_overran_by_one() -> None:
    """Kept as a regression witness -- this is what the code used to do."""
    def old(n_batches, num_iters):
        ran = 0
        for i in range(n_batches):
            ran += 1
            if i == num_iters:
                break
        return ran

    suite.check("the old shape ran one extra iteration", old(400, 3) == 4, str(old(400, 3)))
    suite.check("a 33% overrun on a 3-iteration diagnostic",
                old(400, 3) / 3 > 1.3)
    suite.check("the new shape does not", cap_iterations(400, 3) == 3)


# ---------------------------------------------------------------------------
# 4. the trainer wires all three in
# ---------------------------------------------------------------------------
def test_trainer_exposes_the_new_pieces() -> None:
    import inspect

    from learning import training

    source = inspect.getsource(training.SNNTrainer.train)
    suite.check("train() calls warm_up", "warm_up(" in source)
    suite.check("train() checks the cap before the body",
                "if i >= num_iters:" in source)
    suite.check("the old trailing check is gone", "if i == num_iters:" not in source)
    suite.check("train() resolves the window rather than hardcoding it",
                "firing_window_seconds(" in source)
    suite.check("the phantom attribute is not read anywhere in training",
                "getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US'" not in inspect.getsource(training))


def test_inference_no_longer_reads_phantom_attributes() -> None:
    import inspect

    from learning import inference

    source = inspect.getsource(inference)
    # Matches the READ, not the name -- the name still appears in the comment that
    # explains what went wrong, and that comment is worth keeping.
    suite.check("inference does not read TEMPORAL_SLICE_DURATION_US",
                "getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US'" not in source)
    suite.check("inference does not read cfg.TIMESTEPS",
                "getattr(self.cfg, 'TIMESTEPS'" not in source)
    suite.check("inference resolves the window properly",
                "firing_window_seconds(" in source)
    suite.check("inference reports the time-unit-free figure too",
                "spikes_per_neuron_per_inference" in source)


def test_batch_size_is_recorded(  ) -> None:
    """D11: calibration legitimately picks a different batch size per machine, so the
    value has to travel with the results or two rows cannot be compared on speed."""
    import inspect

    from learning import training

    source = inspect.getsource(training.SNNTrainer.finalize_epoch_reports)
    suite.check("the epoch row records batch_size", '"batch_size"' in source)
    suite.check("and whether it was calibrated", '"batch_size_calibrated"' in source)


# ---------------------------------------------------------------------------
# 5. spike recording is outside every timed region
# ---------------------------------------------------------------------------
class FakeLoader:
    """Stands in for PrefetchedLoader: yields already-device-resident batches."""

    def __init__(self, batches: int, cfg, timesteps: int = 6, batch: int = 2):
        self.batches, self.cfg, self.T, self.B = batches, cfg, timesteps, batch

    def __len__(self):
        return self.batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(0)
        shape = (self.T, self.B, self.cfg.IN_CHANNELS, self.cfg.SENSOR_H, self.cfg.SENSOR_W)
        for _ in range(self.batches):
            yield ((torch.rand(*shape, generator=generator) < 0.5).float(),
                   torch.randint(0, self.cfg.NUM_CLASSES, (self.B,), generator=generator))


def run_short_training(iterations: int = 3, epochs: int = 2, available: int = 10):
    """A real SNNTrainer.train() on CPU with a fake loader. Slower than the rest of
    this suite, but the iteration cap and the activity/timing separation are properties
    of the LOOP -- testing the pieces in isolation would not show them."""
    from learning.training import SNNTrainer

    cfg = fresh_cfg()
    cfg.EPOCHS, cfg.ITERA, cfg.BATCH_SIZE = epochs, iterations, 2
    cfg.ENABLE_PIPELINE_MONITOR = False
    model, _ = build_model("torch", cfg)
    trainer = SNNTrainer(model, FakeLoader(available, cfg), cfg, torch.device("cpu"))
    trainer.train(csv_path=str(REPO_TMP / "unit_training_results.csv"))
    return trainer, model


def test_training_runs_exactly_the_requested_iterations() -> None:
    """D12 end to end: 10 batches available, 3 requested, so every epoch must log 3."""
    trainer, _ = run_short_training(iterations=3, epochs=2, available=10)
    counts = [row["n"] for row in trainer.epoch_log]
    suite.check("every epoch ran exactly the requested iterations", counts == [3, 3],
                str(counts))


def test_recording_is_paused_during_training() -> None:
    trainer, model = run_short_training(iterations=2, epochs=1, available=5)
    suite.check("the monitor is left paused after training", model.activity.paused is True)
    suite.check("no recordings are left dangling",
                all(v is None for v in model.activity.recordings().values()))


def test_activity_metrics_still_land_despite_the_pause() -> None:
    """Pausing must not silently cost the metrics -- that is what the untimed pass is
    for. SynOps and CV_ISI both have to be real, non-zero numbers."""
    trainer, _ = run_short_training(iterations=2, epochs=2, available=5)
    synops = [row["synops_energy_pj"] for row in trainer.epoch_log]
    cv_isi = [row["cv_isi_mean"] for row in trainer.epoch_log]
    suite.check("SynOps is recorded for every epoch", all(v > 0 for v in synops), str(synops))
    suite.check("CV_ISI is recorded for every epoch", all(v > 0 for v in cv_isi), str(cv_isi))


def test_synops_is_per_batch_not_an_iteration_sum() -> None:
    """It used to be summed over every iteration, so it scaled with iteration count and
    two runs of different length could not be compared. Sampling once per epoch means
    doubling the iterations must NOT roughly double it."""
    few, _ = run_short_training(iterations=2, epochs=1, available=10)
    many, _ = run_short_training(iterations=8, epochs=1, available=10)
    a, b = few.epoch_log[0]["synops_energy_pj"], many.epoch_log[0]["synops_energy_pj"]
    suite.check("4x the iterations does not scale SynOps", b < a * 2.0,
                f"2 iters {a:.0f}, 8 iters {b:.0f}")


def test_iteration_series_lengths_line_up() -> None:
    """Every per-iteration series must be the same length as the loss history, or the
    CSV writer's zip() would silently truncate the file."""
    trainer, _ = run_short_training(iterations=3, epochs=2, available=10)
    series = trainer.iteration_series()
    expected = len(trainer.loss_hist)
    suite.check("loss history has one entry per iteration", expected == 6, str(expected))
    for key in ["synops_energy_pj", "gpu_energy_j", "learning_rate",
                "spikes_per_neuron_per_inference", "firing_rate_hz"]:
        suite.check(f"{key} matches the loss history length",
                    len(series[key]) == expected, f"{len(series[key])} vs {expected}")
    suite.check("vram history matches too", len(trainer.vram_current_hist) == expected)


def test_epoch_row_carries_the_new_columns() -> None:
    trainer, _ = run_short_training(iterations=2, epochs=1, available=5)
    row = trainer.epoch_log[0]
    for column in ["energy_j_total", "energy_j_dynamic", "idle_power_w",
                   "spikes_per_neuron_per_inference", "firing_rate_hz",
                   "batch_size", "batch_size_calibrated", "synops_energy_pj"]:
        suite.check(f"epoch row has {column}", column in row)
    suite.check("Hz is None with no stated sample duration", row["firing_rate_hz"] is None,
                str(row["firing_rate_hz"]))


def test_training_runs_on_cpu_at_all() -> None:
    """There was an unguarded torch.cuda.synchronize() at the epoch boundary, so a
    CPU-only torch build died at the end of epoch 1 -- which is every laptop run."""
    try:
        run_short_training(iterations=2, epochs=2, available=5)
        suite.check("a two-epoch CPU run completes", True)
    except AssertionError as error:  # "Torch not compiled with CUDA enabled"
        suite.check("a two-epoch CPU run completes", False, str(error))


def main() -> int:
    return suite.run([
        test_window_is_none_when_not_knowable,
        test_window_from_a_stated_sample_duration,
        test_window_derived_from_time_window_framing,
        test_window_from_temporal_slicing_by_time,
        test_stated_duration_wins_over_derivation,
        test_the_attribute_the_old_code_read_does_not_exist,
        test_spikes_per_inference_is_time_unit_free,
        test_the_size_of_the_old_error,
        test_warmup_leaves_the_weights_untouched,
        test_warmup_clears_gradients,
        test_warmup_leaves_no_activity_recordings,
        test_warmup_zero_is_a_no_op,
        test_warmup_actually_runs_the_requested_count,
        test_warmup_iterations_is_configurable,
        test_iteration_cap_runs_exactly_what_was_asked,
        test_cap_above_the_loader_length_is_harmless,
        test_the_calibrated_case_is_unaffected,
        test_the_old_shape_overran_by_one,
        test_trainer_exposes_the_new_pieces,
        test_inference_no_longer_reads_phantom_attributes,
        test_batch_size_is_recorded,
        test_training_runs_exactly_the_requested_iterations,
        test_recording_is_paused_during_training,
        test_activity_metrics_still_land_despite_the_pause,
        test_synops_is_per_batch_not_an_iteration_sum,
        test_iteration_series_lengths_line_up,
        test_epoch_row_carries_the_new_columns,
        test_training_runs_on_cpu_at_all,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
