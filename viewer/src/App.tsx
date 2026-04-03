import { useState } from "react";

const STATUS = { stable: "#639922", experimental: "#BA7517", critical: "#A32D2D", validated: "#185FA5" };
const STATUS_BG = { stable: "#EAF3DE", experimental: "#FAEEDA", critical: "#FCEBEB", validated: "#E6F1FB" };
const STATUS_LABEL = { stable: "Stable", experimental: "Experimental", critical: "Needs validation", validated: "Validated" };

const phases = [
  {
    id: "dev",
    label: "Development Phase 1",
    color: "#534AB7",
    bg: "#EEEDFE",
    border: "#AFA9EC",
    rdNote: "Foundation layer — all downstream quality depends on decisions made here.",
    blocks: [
      {
        id: "sw", label: "Software track", color: "#534AB7", bg: "#CECBF6", status: "experimental",
        rdStandard: "IEEE 2755 (AI Framework), ISO/IEC 25010 (Software Quality)",
        validationGate: "Unit tests pass >99% on golden spike vectors. Model reproducibility across 3 seeds.",
        items: [
          { label: "Neuron model selection", detail: "Choose between LIF (Leaky Integrate-and-Fire), adaptive LIF, Hodgkin-Huxley, or PLIF. LIF is hardware-efficient; adaptive LIF captures burst dynamics. Document model choice with ablation justification.", rd: "Run ablation: swap neuron model, hold all else constant. Log accuracy delta, spike sparsity, and energy proxy." },
          { label: "PyTorch SNN architecture", detail: "Define layer topology in SpikingJelly or snnTorch. Layers: encoding → hidden spiking layers → decoder. Each layer logs membrane potential and spike rate per batch.", rd: "Architecture search: vary depth (2–6 layers), width (64–512 neurons). Plot accuracy vs. spike sparsity Pareto front." },
          { label: "Spike encoding strategy", detail: "Rate coding (spike count ∝ signal), temporal coding (first-spike latency), or phase coding. Temporal is most energy-efficient for edge deployment.", rd: "Benchmark encoding schemes on event-camera dataset. Metric: classification accuracy / average spike count." },
          { label: "Hyperparameter registry", detail: "Versioned config file (YAML/TOML): weights init scheme, τ_mem, τ_syn, threshold θ, batch size, time window T, learning rate schedule.", rd: "Use Optuna or Ray Tune for Bayesian HPO. Log all trials to MLflow. Never hand-tune without logging." },
          { label: "SIL validation", detail: "Software-in-the-Loop: run identical inference on CPU Python model and compiled CUDA model. Flag any output divergence >0.1% as a regression.", rd: "Regression suite: 500 fixed input vectors. SIL pass is a hard gate before hardware promotion." },
        ]
      },
      {
        id: "hw", label: "Hardware track", color: "#185FA5", bg: "#B5D4F4", status: "critical",
        rdStandard: "MISRA C++ 2023, CUDA Best Practices Guide, ISO 26262 (if safety-critical)",
        validationGate: "Kernel correctness verified against SIL golden vectors. Power within budget ±5%.",
        items: [
          { label: "CUDA kernel architecture", detail: "Each SNN layer maps to a CUDA kernel. Spike events stored as sparse COO tensors. Membrane potential in shared memory. Synaptic weights in L2-cached global memory.", rd: "Profile with Nsight Compute: target >70% SM occupancy, <10% memory stall cycles." },
          { label: "Memory hierarchy design", detail: "Registers → shared mem → L1 → L2 → DRAM. Spike tensors are sparse — use ELL or CSR format to avoid zero-padding. Batch spike events for coalesced global access.", rd: "Roofline model analysis. Plot achieved FLOPS vs. bandwidth. Identify if kernel is compute or memory bound." },
          { label: "Energy efficiency framework", detail: "Clock-gate inactive SMs. Use NVIDIA power capping API. Implement neuron pruning mask to skip zero-spike neurons entirely. Target <1 pJ/spike.", rd: "Measure with nvidia-smi power telemetry. Compare spike-sparse vs dense baseline. Log energy-per-inference as primary KPI." },
          { label: "Noise & robustness model", detail: "Inject Gaussian noise on input spikes (σ=0.01–0.1). Inject bit-flips on weight tensors (BER 1e-6). Validate output degradation is graceful (<3% accuracy drop).", rd: "Monte Carlo noise injection over 1000 trials. Plot accuracy vs. noise level. Define acceptable noise floor." },
          { label: "SM configuration file", detail: "YAML config maps SNN layers to SM blocks: thread block size, warp size, shared mem allocation per block, dynamic parallelism flags.", rd: "Grid search over thread block configs (128, 256, 512 threads). Metric: kernel runtime + occupancy tradeoff." },
          { label: "Clocking & timing analysis", detail: "Static timing analysis on critical paths. Define max clock frequency for target GPU. Validate no race conditions in concurrent kernel streams.", rd: "Use cuda-memcheck and Compute Sanitizer. Zero race conditions is a hard gate." },
        ]
      },
      {
        id: "ctrl", label: "ML control algorithm", color: "#0F6E56", bg: "#9FE1CB", status: "stable",
        rdStandard: "IEC 61508 (Functional Safety), Control loop design patterns",
        validationGate: "Control loop converges within 10 recalibration cycles. No infinite loops.",
        items: [
          { label: "Recalibration trigger logic", detail: "Monitors: accuracy drift >2%, spike rate collapse (<5% sparsity), weight norm explosion (>1e4), kernel timing violation. Any breach fires recalibration.", rd: "Define control chart (CUSUM or Shewhart) for each KPI. Threshold values derived from 10-run baseline distribution." },
          { label: "Matlab/Simulink co-simulation", detail: "Simulink model mirrors the CUDA control loop. Used for formal verification of recalibration logic and timing guarantees before C++ deployment.", rd: "Generate C code from Simulink model. Diff against hand-written C++ control code. Flag divergence." },
          { label: "Fault injection framework", detail: "C++ fault injector: bit-flips (weight/activation), stuck-at-zero neurons, timing violations (+/- 1 clock cycle), memory corruption (single byte). All faults logged.", rd: "Fault coverage metric: % of injected faults detected and recovered. Target >95% fault coverage." },
          { label: "System health dashboard", detail: "Real-time metrics: spike rate per layer, membrane potential distribution, weight norm per layer, recalibration event count, training loss curve.", rd: "Alert thresholds auto-computed from first 5 training runs. Exported as Prometheus metrics." },
        ]
      },
      {
        id: "sim", label: "Simulation", color: "#854F0B", bg: "#FAC775", status: "validated",
        rdStandard: "DO-178C (if avionics), IEEE 1647 (functional verification)",
        validationGate: "Simulation-to-silicon correlation >98% on timing and spike event counts.",
        items: [
          { label: "Cycle-accurate simulator", detail: "Simulates GPU memory hierarchy, SM scheduling, and spike event timing cycle-by-cycle. Validates kernel before real hardware run.", rd: "Correlation test: run identical workload on simulator and real GPU. Flag >2% runtime divergence." },
          { label: "Hardware abstraction layer", detail: "HAL allows same SNN model to target NVIDIA GPU, Intel Loihi 2, IBM TrueNorth, or SpiNNaker 2. Each backend has a verified driver.", rd: "Cross-target regression: same model, all backends, same test set. Document per-backend accuracy and energy." },
          { label: "Event-driven simulation", detail: "For neuromorphic targets: simulate asynchronous spike events with precise timestamps. Validate causality (no effect before cause).", rd: "Causality checker: automated tool that scans event logs for timestamp violations." },
        ]
      },
    ]
  },
  {
    id: "deploy",
    label: "Deployment Phase 2",
    color: "#993556",
    bg: "#FBEAF0",
    border: "#ED93B1",
    rdNote: "Hardware integration layer — where software correctness meets physical constraints.",
    blocks: [
      {
        id: "compile", label: "Compiler pipeline", color: "#993556", bg: "#F4C0D1", status: "stable",
        rdStandard: "LLVM IR verification, nvcc PTX validation",
        validationGate: "Compiled binary matches SIL golden vectors. No UB (undefined behaviour) flags.",
        items: [
          { label: "nvcc / LLVM compilation", detail: "Two-stage: nvcc compiles CUDA → PTX (virtual ISA), then PTX JIT-compiled to SASS (real ISA) at runtime. Enables forward compatibility.", rd: "Compile with -lineinfo for profiling. Always compile with -Werror. PTX diff between versions to catch regressions." },
          { label: "Compiler optimisation flags", detail: "--use_fast_math (approx intrinsics), -O3, --maxrregcount (register pressure), -arch=sm_XX (target SM version). Document all flags in build config.", rd: "A/B test with/without fast_math. Measure accuracy impact. Only enable if delta <0.05%." },
          { label: "Static analysis & linting", detail: "clang-tidy, cppcheck, and CUDA-specific linter. Zero high-severity warnings policy. All warnings treated as errors in CI.", rd: "CI gate: static analysis must pass before merge. Track warning count trend over time." },
          { label: "Thread & stream scheduling", detail: "Map SNN layers to CUDA streams for pipeline parallelism. Overlap data transfer (cudaMemcpyAsync) with compute. Use CUDA graphs for recurring execution patterns.", rd: "Profile stream overlap with Nsight Systems timeline. Quantify latency reduction vs baseline sequential." },
        ]
      },
      {
        id: "gpu", label: "GPU execution", color: "#185FA5", bg: "#B5D4F4", status: "experimental",
        rdStandard: "NVIDIA Best Practices, Roofline model compliance",
        validationGate: "Throughput >X samples/sec (defined per application). Latency <Y ms (defined per SLA).",
        items: [
          { label: "Forward pass execution", detail: "Parallel spike propagation: each SM handles a neuron batch. Membrane potential updated in shared memory. Spike events written to global spike buffer (sparse).", rd: "Benchmark: throughput (samples/s), latency (ms/sample), SM utilisation (%). All three logged per release." },
          { label: "Dynamic precision control", detail: "FP32 for training, FP16 mixed precision for inference, INT8 for edge deployment. Use NVIDIA TensorRT for INT8 calibration with SNN spike tensors.", rd: "Accuracy vs precision tradeoff curve: FP32 → FP16 → INT8. Document acceptable precision floor." },
          { label: "ML threshold decision gate", detail: "After forward pass: check spike rate (target 5–20% sparsity), loss convergence, and inference accuracy. Three-way branch: continue training / recalibrate / promote to portability.", rd: "Gate criteria defined as acceptance test suite. All criteria version-controlled alongside model." },
          { label: "Telemetry & observability", detail: "Per-layer spike histograms, weight norm tracking, GPU utilisation, memory bandwidth usage. All streamed to monitoring backend.", rd: "Define SLOs (Service Level Objectives) for each KPI. Alert if SLO breached for 3 consecutive evaluations." },
        ]
      },
    ]
  },
  {
    id: "train",
    label: "Training & Validation",
    color: "#3B6D11",
    bg: "#EAF3DE",
    border: "#97C459",
    rdNote: "Core learning loop — reproducibility and data integrity are non-negotiable here.",
    blocks: [
      {
        id: "input", label: "Input pipeline", color: "#3B6D11", bg: "#C0DD97", status: "experimental",
        rdStandard: "ISO/IEC 25012 (Data Quality), GDPR if personal data",
        validationGate: "Dataset statistics (mean, variance, class balance) logged and version-locked per experiment.",
        items: [
          { label: "Event camera pipeline", detail: "DVS sensor → event denoising (hot pixel filter, refractory filter) → event-to-frame conversion (voxel grid or time surface) → spike tensor. Timestamp precision: microsecond.", rd: "Validate denoising: compare signal-to-noise ratio before/after filter on synthetic events with known ground truth." },
          { label: "Data stream ingestion", detail: "Continuous analogue streams encoded as spike trains. Rate coding: spike count proportional to signal amplitude. Temporal: first-spike time inversely proportional to amplitude.", rd: "Encoding fidelity test: reconstruct original signal from spike train. Measure RMSE vs original." },
          { label: "Dataset versioning", detail: "DVC (Data Version Control) or MLflow Artifacts. Every experiment links to exact dataset version, preprocessing config, and random seed.", rd: "Reproducibility check: re-run experiment with same dataset version and seed. Expect <0.1% accuracy variance." },
          { label: "Data augmentation", detail: "Spike jitter (±1 timestep), random neuron dropout (5–15%), polarity flip (event cameras), temporal rescaling (0.8–1.2×). All augmentations logged.", rd: "Ablation: train with/without each augmentation. Keep only augmentations that improve val accuracy." },
        ]
      },
      {
        id: "trainblock", label: "Training loop", color: "#1D9E75", bg: "#5DCAA5", status: "critical",
        rdStandard: "MLflow experiment tracking, Model Cards (Google), Datasheets for Datasets",
        validationGate: "Training loss converges (<1% delta over last 10 epochs). No NaN/Inf in gradients.",
        items: [
          { label: "Surrogate gradient BPTT", detail: "Straight-through estimator or Sigmoid surrogate for spike non-differentiability. BPTT unrolled over T timesteps. Gradient clipping at norm 1.0 to prevent explosion.", rd: "Compare surrogate functions (STE, sigmoid, triangle, arctan). Plot gradient magnitude distribution per layer." },
          { label: "STDP local learning", detail: "Spike-Timing Dependent Plasticity as unsupervised pre-training. Potentiation if pre-synaptic spike precedes post (Δt < 0), depression if post precedes pre.", rd: "STDP + fine-tune vs pure supervised: compare final accuracy and convergence speed." },
          { label: "Loss function design", detail: "Options: rate-coded cross-entropy, Van Rossum spike distance, membrane potential MSE, or hybrid. Loss must be differentiable w.r.t. continuous surrogate.", rd: "Loss landscape visualisation (filter normalisation method). Identify saddle points or sharp minima." },
          { label: "Experiment tracking", detail: "MLflow: log hyperparams, metrics per epoch, model checkpoints, confusion matrix, spike raster plots, environment (GPU, CUDA version, library versions).", rd: "Every experiment must be reproducible from MLflow run ID alone. Periodic audit: pick random past run, reproduce it." },
          { label: "Regularisation strategy", detail: "L1 on weights (promotes sparsity), spike rate regularisation (penalise firing rate outside 5–20%), dropout on spiking layers (p=0.1–0.2).", rd: "Regularisation ablation: train with each term isolated. Measure sparsity, accuracy, energy proxy trade-off." },
        ]
      },
      {
        id: "infer", label: "Inference & validation", color: "#0F6E56", bg: "#9FE1CB", status: "validated",
        rdStandard: "ISO/IEC 29119 (Software Testing), Model evaluation best practices",
        validationGate: "Accuracy on held-out test set meets spec. Latency SLA met. Energy budget met.",
        items: [
          { label: "Inference benchmarking", detail: "Latency (p50, p95, p99 ms), throughput (samples/s), energy (mJ/sample), spike sparsity (%), accuracy (%). All five metrics required for promotion.", rd: "Benchmark on 3 hardware configs: training GPU, edge GPU (Jetson), target neuromorphic. Document all." },
          { label: "Confusion matrix analysis", detail: "Per-class precision, recall, F1. Identify systematically confused classes. Link confusion to spike pattern analysis (which neurons fire for confused classes).", rd: "Error analysis sprint: for top-3 confused class pairs, trace back to input encoding and layer activations." },
          { label: "Adversarial robustness test", detail: "FGSM and PGD attacks on spike inputs. Measure accuracy drop. SNN inherent robustness from spike discretisation — quantify this advantage.", rd: "Compare adversarial robustness: SNN vs equivalent ANN baseline on same task. Publish delta as R&D claim." },
          { label: "ML decision gate", detail: "Structured checklist: accuracy ≥ spec ✓, latency ≤ SLA ✓, energy ≤ budget ✓, sparsity in range ✓, no regressions vs previous model ✓. All five must pass.", rd: "Gate is automated and version-controlled. Manual override requires sign-off and written justification." },
        ]
      },
    ]
  },
  {
    id: "port",
    label: "Portability",
    color: "#444441",
    bg: "#F1EFE8",
    border: "#B4B2A9",
    rdNote: "Product readiness layer — transforms a working model into a deployable product.",
    blocks: [
      {
        id: "extract", label: "NN extraction", color: "#444441", bg: "#D3D1C7", status: "stable",
        rdStandard: "ONNX spec, MLflow Model Registry, SemVer model versioning",
        validationGate: "Exported model inference output matches source model output within floating point tolerance.",
        items: [
          { label: "Weight serialisation", detail: "Export to ONNX (cross-framework), HDF5 (legacy), or custom SNN binary format. Include: weight tensors, threshold values, τ_mem, τ_syn per layer.", rd: "Round-trip test: export → import → infer. Compare outputs. Flag any divergence." },
          { label: "Graph pruning & quantisation", detail: "Prune neurons with <1% average firing rate. Fold batch-norm. Post-training quantisation to INT8 using SNN-specific calibration (spike count histograms).", rd: "Pruning sensitivity analysis: measure accuracy drop per pruning ratio. Find Pareto-optimal sparsity point." },
          { label: "Model card generation", detail: "Automated Model Card: intended use, training data description, evaluation results (all 5 metrics), known failure modes, energy characterisation, version history.", rd: "Model Card is a release artefact. Reviewed by team before each deployment. Version-locked with model weights." },
        ]
      },
      {
        id: "portbuild", label: "Target build", color: "#3C3489", bg: "#CECBF6", status: "experimental",
        rdStandard: "IPC-7711 (hardware rework), DO-254 (FPGA if avionics)",
        validationGate: "Binary boots on target device. Smoke test passes within 5 minutes of flash.",
        items: [
          { label: "FPGA mapping (HLS)", detail: "Vivado HLS or Intel HLS: C++ SNN description → RTL. Map neuron groups to FPGA BRAMs. Implement spike routing as AXI-Stream pipeline.", rd: "Resource utilisation report: LUT%, BRAM%, DSP%. Target <80% utilisation for thermal headroom." },
          { label: "Neuromorphic target (Loihi 2)", detail: "Intel NxSDK: map SNN graph to Loihi 2 neurocore mesh. Each neurocore handles 1024 compartments. Spike routing via on-chip mesh network.", rd: "Compare Loihi 2 vs GPU: energy/inference, latency, accuracy. Document crossover point (batch size where GPU wins)." },
          { label: "Edge GPU (Jetson)", detail: "TensorRT engine build from ONNX. INT8 calibration with spike tensor dataset. CUDA streams for pipeline overlap. Power mode: MaxN vs 10W budget mode.", rd: "Jetson power profiling: tegrastats during inference. Compare MaxN vs 10W accuracy-latency-power trade-off." },
          { label: "Cross-compilation pipeline", detail: "CMake cross-compile toolchain file per target. Docker-based build environment (pinned versions). Artefacts signed with SHA256. CI builds all targets on every release branch commit.", rd: "Build matrix in CI: all targets × all configs × smoke test. Any failure blocks release." },
        ]
      },
      {
        id: "hwdeploy", label: "Hardware deployment", color: "#993C1D", bg: "#F5C4B3", status: "validated",
        rdStandard: "IEC 62443 (industrial security), FMEA before first deployment",
        validationGate: "All acceptance tests pass on physical hardware. Power within budget. FMEA completed.",
        items: [
          { label: "FMEA pre-deployment", detail: "Failure Mode and Effects Analysis: enumerate all failure modes (power loss, spike flood, weight corruption, thermal throttle). For each: severity, probability, detectability. Mitigate all S×P×D > threshold.", rd: "FMEA completed and signed off before first hardware flash. Reviewed at every major version bump." },
          { label: "Flash & acceptance test", detail: "Automated flash script → reboot → run acceptance test suite (500 samples, known labels). Pass criterion: accuracy ≥ spec, latency ≤ SLA, no crash in 10-min soak.", rd: "Acceptance test is immutable and version-locked with firmware. Results logged to deployment database." },
          { label: "Runtime telemetry", detail: "On-device: spike rate per layer, inference latency histogram, temperature, power draw (INA219 or platform API). Streamed to central monitoring (Prometheus + Grafana).", rd: "SLO monitoring: alert if p95 latency > 2× baseline or accuracy drift > 3% over rolling 1000-sample window." },
          { label: "OTA update mechanism", detail: "Signed firmware packages (Ed25519). Staged rollout: 1% → 10% → 100%. Automatic rollback if acceptance tests fail post-update.", rd: "Chaos test: simulate update failure mid-flash. Verify automatic rollback restores previous working firmware." },
          { label: "Field recalibration trigger", detail: "If telemetry detects distribution shift (input spike statistics diverge from training distribution), alert sent → human review → optional recalibration pipeline trigger.", rd: "Define distribution shift detector (e.g. MMD or KL divergence on spike histograms). Tune sensitivity on held-out shift dataset." },
        ]
      },
    ]
  },
];

const Badge = ({ status }: { status: string }) => (
  <span style={{
    background: (STATUS_BG as Record<string, string>)[status], color: (STATUS as Record<string, string>)[status],
    fontSize: 11, fontWeight: 500, padding: "2px 8px",
    borderRadius: 6, border: `0.5px solid ${(STATUS as Record<string, string>)[status]}`, whiteSpace: "nowrap"
  }}>{(STATUS_LABEL as Record<string, string>)[status]}</span>
);

export default function App() {
  const [expandedPhase, setExpandedPhase] = useState<string>("dev");
  const [activeBlock, setActiveBlock] = useState<string | null>(null);
  const [activeItem, setActiveItem] = useState<number | null>(null);
  const [tab, setTab] = useState<string>("detail");

  const currentPhase = phases.find(p => p.id === expandedPhase);
  const currentBlock = currentPhase?.blocks.find(b => b.id === activeBlock);

  const handlePhase = (id: string) => {
    setExpandedPhase(id);
    setActiveBlock(null);
    setActiveItem(null);
  };

  const handleBlock = (id: string) => {
    setActiveBlock(id === activeBlock ? null : id);
    setActiveItem(null);
    setTab("detail");
  };

  return (
    <div style={{ fontFamily: "var(--font-sans)", padding: "1.25rem 1rem", maxWidth: 740, margin: "0 auto" }}>

      {/* Phase nav */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: "1.25rem", flexWrap: "wrap" }}>
        {phases.map((ph, i) => (
          <div key={ph.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <button onClick={() => handlePhase(ph.id)} style={{
              background: expandedPhase === ph.id ? ph.color : "var(--color-background-secondary)",
              color: expandedPhase === ph.id ? "#fff" : "var(--color-text-primary)",
              border: `1px solid ${expandedPhase === ph.id ? ph.color : "var(--color-border-secondary)"}`,
              borderRadius: 8, padding: "5px 12px", fontSize: 13, fontWeight: 500, cursor: "pointer",
            }}>{ph.label}</button>
            {i < phases.length - 1 && <span style={{ color: "var(--color-text-tertiary)", fontSize: 14 }}>→</span>}
          </div>
        ))}
      </div>

      {currentPhase && (
        <div style={{ border: `1px solid ${currentPhase.border}`, borderRadius: 12, padding: "1rem", background: "var(--color-background-primary)", marginBottom: "1rem" }}>

          {/* Phase header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem", flexWrap: "wrap", gap: 6 }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: currentPhase.color, letterSpacing: .3 }}>{currentPhase.label.toUpperCase()}</div>
            <div style={{ fontSize: 11, color: "var(--color-text-secondary)", fontStyle: "italic", maxWidth: 420 }}>{currentPhase.rdNote}</div>
          </div>

          {/* Block cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(155px, 1fr))", gap: 8, marginBottom: "0.75rem" }}>
            {currentPhase.blocks.map(bl => (
              <div key={bl.id} onClick={() => handleBlock(bl.id)} style={{
                background: activeBlock === bl.id ? bl.bg : "var(--color-background-secondary)",
                border: `1px solid ${activeBlock === bl.id ? bl.color : "var(--color-border-tertiary)"}`,
                borderRadius: 10, padding: "10px 12px", cursor: "pointer", transition: "all .15s",
              }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: activeBlock === bl.id ? bl.color : "var(--color-text-primary)", marginBottom: 6 }}>{bl.label}</div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{bl.items.length} components</span>
                  <Badge status={bl.status} />
                </div>
              </div>
            ))}
          </div>

          {/* Block detail */}
          {currentBlock && (
            <div style={{ background: "var(--color-background-secondary)", borderRadius: 10, padding: "1rem", borderLeft: `3px solid ${currentBlock.color}` }}>

              {/* Block meta */}
              <div style={{ marginBottom: "0.75rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 14, fontWeight: 500, color: currentBlock.color }}>{currentBlock.label}</span>
                  <Badge status={currentBlock.status} />
                </div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 2 }}>
                  <strong style={{ color: "var(--color-text-primary)" }}>Standard: </strong>{currentBlock.rdStandard}
                </div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>
                  <strong style={{ color: "var(--color-text-primary)" }}>Validation gate: </strong>{currentBlock.validationGate}
                </div>
              </div>

              {/* Tab bar */}
              <div style={{ display: "flex", gap: 6, marginBottom: "0.75rem" }}>
                {["detail", "rd"].map(t => (
                  <button key={t} onClick={() => setTab(t)} style={{
                    background: tab === t ? currentBlock.color : "transparent",
                    color: tab === t ? "#fff" : "var(--color-text-secondary)",
                    border: `0.5px solid ${tab === t ? currentBlock.color : "var(--color-border-secondary)"}`,
                    borderRadius: 6, padding: "3px 10px", fontSize: 12, cursor: "pointer",
                  }}>{t === "detail" ? "Components" : "R&D protocol"}</button>
                ))}
              </div>

              {/* Items */}
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {currentBlock.items.map((item, i) => (
                  <div key={i} onClick={() => setActiveItem(activeItem === i ? null : i)} style={{
                    background: "var(--color-background-primary)",
                    border: `0.5px solid ${activeItem === i ? currentBlock.color : "var(--color-border-tertiary)"}`,
                    borderRadius: 8, padding: "10px 12px", cursor: "pointer",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>{item.label}</span>
                      <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>{activeItem === i ? "▲" : "▼"}</span>
                    </div>
                    {activeItem === i && (
                      <div style={{ marginTop: 8, fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.7 }}>
                        {tab === "detail" ? item.detail : (
                          <span>
                            <span style={{ display: "inline-block", background: STATUS_BG.validated, color: STATUS.validated, fontSize: 11, padding: "1px 7px", borderRadius: 5, marginRight: 6, fontWeight: 500 }}>R&D</span>
                            {item.rd}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: "0.75rem" }}>
        {Object.entries(STATUS_LABEL).map(([k, v]) => (
          <span key={k} style={{ background: (STATUS_BG as Record<string, string>)[k], color: (STATUS as Record<string, string>)[k], fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 6, border: `0.5px solid ${(STATUS as Record<string, string>)[k]}` }}>{v}</span>
        ))}
        <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", alignSelf: "center" }}>— R&D readiness status per block</span>
      </div>

      {/* Recalibration loop */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: "12px 14px", fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
        <span style={{ fontSize: 15, marginTop: 1 }}>↻</span>
        <div>
          <span style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>Global recalibration loop: </span>
          Triggered by: accuracy drift &gt;2%, spike rate collapse, telemetry SLO breach, distribution shift detection, or field FMEA event.
          Routes back to Development Phase 1 with a structured incident report: failure mode, affected layers, proposed fix, and required validation gate re-run.
          All recalibration events logged to the experiment tracker with full context.
        </div>
      </div>

      <div style={{ marginTop: "0.6rem", fontSize: 11, color: "var(--color-text-tertiary)", textAlign: "center" }}>
        Click phase → block → component · Toggle Components / R&D protocol tabs
      </div>
    </div>
  );
}
