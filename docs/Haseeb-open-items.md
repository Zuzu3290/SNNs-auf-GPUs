# To DO

- different prams for different frameworks like Threshold vs v_th  in snn_torch vs norse. Implement them either in config as frameworks blocks or somre mapping , example beta has range 0-1, and tau: 10-1000 Hz even they have same meaning, found beta nad tau as clashing so far.

---

- No loss function in Norse:
Norse has no built-in loss functions — it only provides neuron dynamics (LIF, etc.) and leaves loss entirely to you. So the options are:

Option 1 — Keep SF.mse_count_loss (what we did)
It's purely tensor math operating on [T, B, num_classes] — it doesn't use any SNNTorch neuron internals. Norse produces the same shaped spike tensor, so it works fine.

Option 2 — Standard PyTorch CrossEntropy on summed spikes


loss_fn = torch.nn.CrossEntropyLoss()
# usage: loss_fn(spk_rec.sum(0), targets)  # sum over T → [B, num_classes]
For this project specifically, Option 1 is actually the right choice. You're comparing frameworks — if both models use the exact same loss function, you isolate the variable to the neuron dynamics only. Switching loss functions between SNNTorch and Norse would make the comparison unfair.

---





# Improevements in Future

- TUM-VIE (Phase B regression dataset) — mocap ground truth is only captured at the
  beginning and end of each recording (TUM's own docs), not continuously through it.
  So it can't supervise a full-sequence trajectory the way we first assumed. Follow-up
  idea, not built yet: reframe as a net-displacement regression — take a bounded
  window near a recording's start/end where mocap brackets both sides, predict the
  single (Δposition, Δorientation) between the two anchors from the event stream in
  between, instead of a dense per-timestep pose. Also worth a second look for other
  task framings later (it has stereo events + IMU + grayscale frames too, not just
  mocap) rather than writing it off entirely.

- Regression model variants now exist (frameworks/personal/snn_*_regression.py, gitignored/
  local-only) for all 4 active backends (Norse, snnTorch, SpikingJelly, Sinabs) — same conv+LIF
  backbone as the classification models, but a plain linear readout (cfg.REGRESSION_OUTPUT_DIM,
  default 6) instead of a spiking classification head, MSE loss instead of cross-entropy.
  Verified in isolation: each builds, forwards, computes loss, and backprops correctly against
  dummy tensors. main.py now dispatches to these when a dataset's task_type is "regression" and
  the framework has a variant (see _REGRESSION_MODELS in main.py).

  Still blocking real end-to-end training on MVSEC/TUM-VIE — two concrete, separate gaps:
  1. `SNNTrainer.train()` (learning/training.py) does `targets.to(self.device, non_blocking=True).long()`
     unconditionally — assumes targets is already a plain tensor. MVSEC/TUM-VIE's collated
     targets are a Python list of raw dicts/tuples (event_data_workflow/data_pipeline.py's
     pad_events_passthrough_target passes them through unchanged, since PadTensors' own
     torch.tensor(target) call crashes on them). Needs a target-extraction adapter that pulls a
     single (B, REGRESSION_OUTPUT_DIM) tensor out of that list before the trainer touches it —
     which needs decision #2 below to know what to extract.
  2. Which field becomes the training target isn't decided yet: MVSEC needs depth vs. pose vs. a
     derived flow (see the MVSEC entry earlier in this file); TUM-VIE needs the net-displacement/
     windowing reframing above. Until one is picked, there's nothing concrete for the adapter in
     #1 to extract.

  SNNTester/AdversarialEvaluator also assume classification-shaped results (overall_accuracy,
  confusion matrix, etc.) and haven't been touched — same reason, blocked on #2 first.

  docs/results/run_benchmark.py + make_plots.py were generalized to accept --dataset for any of
  the 6 datasets (not just N-MNIST) and already pick REGRESSION_MODELS automatically for MVSEC/
  TUM-VIE — but running the benchmark against either will still fail inside SNNTrainer for the
  same reason as #1 above, until the adapter exists.



# Found Issue --- Status