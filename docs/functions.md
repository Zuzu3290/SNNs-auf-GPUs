# Function & Call-Site Design Notes

Design rationale for functions/call orderings that aren't obvious from the
function's own docstring — usually because the "why" lives at the call site
(learning/main.py) rather than inside the function itself.

## `calibrate_batch_size` (learning/utilities.py) — call-site ordering

Called from `learning/main.py`, before `NeuromorphicEncoder` builds any
DataLoader:

```python
cfg.apply_dataset_shape(sensor_h=sensor_h, sensor_w=sensor_w, in_channels=in_channels,
                         num_classes=dataset_entry["num_classes"])
cfg.BATCH_SIZE = calibrate_batch_size(ModelClass, cfg, device)

encoder = NeuromorphicEncoder(cfg)
train_loader, test_loader = encoder.get_dataloaders()
```

- `cfg.BATCH_SIZE` must be set before the encoder builds `train_loader` and
  `test_loader`, since both loaders read it at construction time.
- One calibration policy covers both phases — there's no separate
  inference batch size. Inference's real VRAM footprint is smaller than
  training's (no backward graph, no gradients, no optimizer state), so a
  batch size that fits training's larger footprint fits inference too.

## `select_inference_mode` (learning/utilities.py)

Prompts the user, right before the inference/testing phase, to choose
between statistics-only output and statistics + a live matplotlib
visualization of input frames vs. predictions.

- Prints a numbered picker and reads one line via `input()`.
- No stdin attached (Colab, CI, batch — having a terminal to launch the
  process doesn't guarantee stdin is a live keyboard) raises `EOFError`,
  caught to default to statistics-only rather than crashing the run over
  a cosmetic choice.
- An invalid or empty selection also falls back to statistics-only rather
  than erroring, since this is a cosmetic choice, not a config error.

## `SystemResourceMonitor.dataloader_config` (event_data_workflow/system_monitor.py)

Turns live system state into the actual `DataLoader` construction policy —
this is the decision point for how data physically moves from disk/RAM into
the GPU during a run: how many CPU worker processes read and transform
samples in parallel, how many batches each worker stages ahead of time
(`prefetch_factor`), whether host memory is pinned for faster CPU→GPU
copies (`pin_memory`), and whether worker processes are kept alive across
epochs instead of respawned (`persistent_workers`). Called from
`NeuromorphicEncoder.create_loaders()` as `monitor.dataloader_config(self.cfg, device)`.

- Reads live state via `self.snapshot()` (available RAM) and `os.cpu_count()`
  (logical CPU count) — the same snapshot mechanism every other resource
  decision in this codebase uses (e.g. `AdaptiveCacheController`'s cache-mode
  choice in `cache_engine.py`), rather than a separate probe.
- GPU-only fallback: if the RAM budget left over for worker processes drops
  under 500MB, workers are disabled entirely (`num_workers=0`) — an
  embedded/GPU-only run has no host RAM headroom for multiprocessing workers,
  and forcing workers on anyway would OOM the host, not just slow it down.
- Otherwise, worker count is capped by both the RAM budget (bytes available
  ÷ estimated per-batch size) and the machine's actual logical CPU count,
  whichever is smaller — never requests more parallelism than the hardware
  or the RAM budget can actually support.
- Logs a warning (not an error) when the chosen worker count is under half
  the machine's CPU count, since that's a common silent cause of the GPU
  sitting idle waiting on CPU-side data preparation.
