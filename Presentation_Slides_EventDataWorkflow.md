# This Week's Presentation — Event Data Workflow (10-12 min)

## Before the slides: what this presentation is actually claiming

This talk is scoped to **two** of Prof. Bauer's questions, using only the
`event_data_workflow/` pipeline (the data-loading/caching engine, not the
framework comparison — that's a separate piece of work):

1. **Comparative Analysis — the "performance" axis.** Not framework vs.
   framework, but "is the pipeline itself efficient" — measured, not assumed.
   Real evidence: `diagnostics/gpu_performance_investigation_report.md`
   found the pipeline wasting most of its time waiting for data, fixed it, and
   measured a **2.6x** speedup with the wait-time itself cut **105x**.
2. **User Story D (the large-scale researcher)** — what breaks when the
   problem gets bigger. Real evidence: `event_data_workflow/vram_batch_scaling_task.md`
   found and explained an actual crash — memory use exploding with camera
   resolution, not neuron count — confirmed by a real out-of-memory error, not
   a guess.

The connecting thread across both (and a clean way to open with "the Why"):
event cameras report changes as they happen — asynchronously — but every
tool that trains these networks on a GPU (ours included) converts that into
fixed time-slices first. That conversion is a real design choice with a real
cost, and it's the concrete version of the "trade-off" the professor's
message hints at, rather than an abstract statement about it.

If you intended a different "two questions," swap the objective slides (5-10
below) — the title/why/closing slides still work as scaffolding either way.

**Everything on the actual slides below is written in plain language, no
ML/CS jargon** — per your standing instruction that presentation slides stay
readable to a non-technical audience even when the underlying work is
technical. Technical terms only appear in this file's own notes to you, never
in the slide text itself.

---

## Image sourcing rule — read this before generating anything

Two different kinds of images appear below. Don't mix them up:

- **REAL images** (marked `[USE REAL FILE]`) — actual measured results or
  actual camera data already sitting in `diagnostics/`. Use the file
  directly. Do **not** regenerate these with AI — an AI-generated chart would
  be inventing data that looks like evidence but isn't.
- **CONCEPT images** (marked `[NANO BANANA PROMPT]`) — illustrations that
  explain an idea (a trade-off, a comparison, a process), where nothing real
  needs to be plotted. These are safe and appropriate to generate.

---

## Slide 1 — Title

**Type:** Graphic (full bleed)

**On-slide text:**
> Teaching a Computer to See Like an Event Camera
> How our data pipeline performs, and where it breaks

**[NANO BANANA PROMPT]**
```
A clean, modern presentation title background. Abstract visualization of a
silicon retina / event camera: a dark navy background with sparse, glowing
cyan and white dots scattered unevenly across the frame, as if only moving
edges of a scene are lit up, the rest in darkness. No text, no logos, no
people. Minimalist, high-contrast, slightly technical but elegant — suitable
as a title-slide background with text overlaid on the left third. 16:9,
flat/vector illustration style, no photorealism.
```

---

## Slide 2 — The real-world question

**Type:** Text

**On-slide text:**
- Normal cameras record full pictures, many times per second — most of that
  is wasted, because most of a scene doesn't change frame to frame.
- Event cameras only report what changed, the instant it changed. Far less
  data, far less wasted effort — closer to how our own eyes work.
- The catch: that data doesn't arrive in neat, evenly-spaced pictures. It
  arrives as a stream of "something changed here, now" — and every tool we
  train on a graphics card needs data in neat pictures.
- **Our question: what does it cost to bridge that gap, and where does it
  break?**

*(No image needed — keep this one text-only so the idea lands before any
visuals start.)*

---

## Slide 3 — Why there's no single "right" tool

**Type:** Graphic (concept)

**On-slide text:**
- Every simulation tool for this kind of brain-inspired computing has to
  choose between three things it can't all have at once:
  1. How closely it copies real brain cells
  2. How closely it matches the exact chip it will eventually run on
  3. How fast it can actually run today, on hardware we have
- Pick two. No tool gets all three.

**[NANO BANANA PROMPT]**
```
A simple, elegant triangle diagram for a presentation slide, flat vector
illustration style, white or very light background. Three corners of an
equilateral triangle, each with a small simple icon and a short label below
it: top corner — a stylized brain/neuron icon, label "Copies real brain
cells"; bottom-left corner — a stylized computer chip icon, label "Matches
real hardware"; bottom-right corner — a stylized stopwatch/lightning bolt
icon, label "Runs fast today". Thin grey lines connecting the three corners.
No other text. Muted, professional color palette (soft blue, soft teal, soft
orange for the three icons). 16:9 or 4:3, plenty of empty space around the
triangle for a slide layout.
```

---

## Slide 4 — What we actually built

**Type:** Mixed (text + real images)

**On-slide text:**
- A pipeline that takes raw event-camera recordings, stores them
  efficiently, and slices them into consistent time windows a training
  system can actually use.
- It automatically decides *how* to store each dataset — in fast memory, on
  disk, or on the graphics card itself — based on what the machine running it
  actually has available at that moment.
- Built and tested against real recordings from several different event
  cameras, not just one toy example.

**[USE REAL FILE(S)]** — pick 2-3 for a small side-by-side grid:
```
diagnostics/samples/DVS128_Gesture/sample_0.png
diagnostics/samples/N-Caltech101/sample_0.png
diagnostics/samples/DSEC/sample_0.png
```
*(These are real decoded frames from real event-camera datasets — use them
as-is, arranged as a 3-panel strip along the bottom or right of the slide.)*

---

## Slide 5 — Objective 1: Does it actually keep up?

**Type:** Text

**On-slide text:**
- A graphics card is only doing useful work when it's actually computing —
  every second spent waiting for data is a wasted second.
- **The question we tested:** once training is running, how much of the
  graphics card's time is spent waiting instead of working?
- We built our own measurement tool to watch this happen in real time,
  rather than guessing from a single overall speed number.

---

## Slide 6 — Result: we found real waste

**Type:** Graphic (real result)

**On-slide text:**
- Before any fix: the graphics card spent the large majority of each step
  just waiting for the next piece of data.
- This wasn't a guess — it's measured, second by second, while the pipeline
  actually ran.

**[USE REAL FILE]**
```
diagnostics/gpu_utilization_report_baseline.png
```
*(Real measured chart — the "before" picture. Don't relabel or redraw it;
just crop/resize to fit the slide.)*

---

## Slide 7 — Result: we fixed it, and measured the fix

**Type:** Graphic (real result)

**On-slide text:**
- We changed how data is handed to the graphics card — preparing the next
  piece of work while the current one is still being computed, instead of
  one at a time.
- **Result: the time spent waiting dropped by over 100x. Total time per
  training step dropped by more than half.**

**[USE REAL FILE]**
```
diagnostics/gpu_utilization_report_optimized.png
```
*(Real measured chart — the "after" picture. Shown next to Slide 6's image,
the visual difference alone makes the point before you even say the
numbers.)*

---

## Slide 7a — Diagnostics: What We Measured

**Type:** Text

**On-slide text:**
- We tracked three things, every single step of training: time spent waiting
  for data, time spent moving that data onto the graphics card, and time
  spent actually computing.
- We watched the graphics card's real activity while this happened — a live
  measurement, sampled about 20 times a second, not an estimate.
- We ran the exact same pipeline twice, with only one thing changed, so the
  improvement can be pinned on that one change and nothing else.
- **Headline: over 100x less time spent waiting. More than twice as fast
  overall, step for step.**

**Speaker notes (say, don't put on slide):** The three tracked phases are
"fetch" (data-loading/cache work), "transfer" (CPU→GPU copy), "compute"
(the real forward+backward pass) — measured via NVML polling every ~50ms.
The two runs compared are "baseline" vs. "optimized" (async prefetching on).
The first batch of each run is excluded from the averages — it carries a
one-time worker-startup cost that never repeats, and including it would
understate the real steady-state gain. Numbers behind the headline: fetch
216.53ms → 2.07ms per batch (105x), total time per batch 349.82ms → 134.90ms
(2.6x). Full data in `diagnostics/gpu_utilization_per_batch_*.csv` if asked.

---

## Slide 8 — Objective 2: What happens when things get bigger?

**Type:** Text

**On-slide text:**
- Real deployments won't always use small, simple cameras. Higher-resolution
  event cameras, or larger networks of simulated brain cells, are the
  realistic target.
- **The question we tested:** does the pipeline that works fine on a small
  example keep working when the camera — or the network — gets bigger?

---

## Slide 9 — Result: we hit a wall, and know exactly why

**Type:** Graphic (concept — illustrating a real, already-measured finding)

**On-slide text:**
- Switching to a higher-resolution camera — about 46x more detail per frame
  — crashed the system outright. It ran out of memory.
- We traced the exact cause: training a brain-inspired network means holding
  *every* moment in time in memory at once, not just the final answer. That
  cost grows with camera detail much faster than most people expect.
- Cutting how much data we process at once made it work again — but that's a
  trade, not a fix: doing less at a time to fit in memory.

**[NANO BANANA PROMPT]**
```
A simple before/after comparison illustration for a presentation slide, flat
vector style, light background. Two stacked bars side by side under the
labels "Small camera" and "Large camera" — the "Small camera" bar is short
and colored calm blue with a small checkmark icon above it; the "Large
camera" bar is much taller (roughly 8-10x the height of the first, exceeding
a dashed horizontal line labeled "Available memory"), colored warm red/orange
with a small warning triangle icon above it. Minimal, clean, no extra text
beyond the two labels and the dashed "Available memory" line. Professional
muted color palette, plenty of white space, 4:3 or 16:9.
```

*(Note: the "~46x" and the exact crash/fix details are real, measured numbers
from `event_data_workflow/vram_batch_scaling_task.md` — say them out loud
from the notes below the slide rather than baking exact figures into the
image, so the illustration stays simple.)*

**Speaker notes (say, don't put on slide):** Conv1 layer output goes from
10,800 numbers per image at small resolution to 498,432 at large resolution.
Holding every one of those, for every moment in time, for every item in a
batch, is what actually exhausts memory — confirmed by literally hitting an
out-of-memory crash at the original settings, and confirming it runs cleanly
once we reduce how much we process at once.

---

## Slide 9a — Bigger Cameras: What We Measured, and a Second Problem

**Type:** Text

**On-slide text:**
- What we tracked: how detailed the camera image is, how many samples are
  processed together, and how many moments in time are held in memory at
  once.
- We predicted the memory cost by hand from those numbers first, then
  checked it against a real run — the prediction and the real crash matched.
  That match is why we trust the explanation, not just the crash itself.
- **A second, separate problem, found along the way: how many samples to
  process together was set as one fixed number for every camera — not
  adjusted for how demanding each individual camera actually is.**
- **Already solved:** the network's shape — image size in, number of
  categories out — already adjusts automatically for whichever camera is
  used. No manual changes needed there, confirmed working across every
  camera we support.
- **Still open, not yet solved:** doing that same automatic adjustment for
  how many samples to process at once. The same system already has a slot
  ready for exactly this, per camera — it just hasn't been switched on yet.

**Speaker notes (say, don't put on slide):** The shape adaptation is
`apply_dataset_shape()` — real, tested, wired for every dataset in the
registry. The batch-size slot is the same registry's `batch_size` field per
dataset entry (`None` = inherit the global default) — the mechanism
(`apply_dataset_hyperparams()`) is real and already wired through, but
N-Caltech101's own entry is still `None`, so it still silently inherits the
global batch size that crashes it. If asked directly: this is a known,
scoped, one-line fix, not yet applied — say that plainly rather than
implying it's done. (Documented in
`event_data_workflow/vram_batch_scaling_task.md` and
`dataset_registry.py`.)

---

## Slide 10 — Why this matters beyond our project

**Type:** Text

**On-slide text:**
- Anyone deploying this kind of system on a real, higher-resolution camera —
  robotics, self-driving perception, security — will hit this same memory
  wall, not a speed wall, first.
- Anyone trying to simulate very large networks of brain-like cells hits the
  same root cause from a different direction: more cells, more memory held
  at once.
- **Our contribution here isn't "we solved large-scale simulation" — it's
  "we found and explained the actual wall you hit first, with a real crash
  to prove it," which is a more useful starting point than a guess.**

---

## Slide 11 — Closing

**Type:** Text

**On-slide text:**
- Two concrete, measured results this week:
  1. Found and fixed a real inefficiency — over 100x less time wasted
     waiting for data, more than 2x faster overall.
  2. Found and explained a real limit — memory, not speed, is what breaks
     first as cameras or networks get bigger.
- Both came from building our own measurement tools and watching the system
  run, not from assumptions.
- Next stage of the project goes deeper into the hardware side of this
  question — more on that in the next presentation.

*(Keep the "next stage" line vague on purpose — the Brian2CUDA plan isn't
part of this week's scope.)*

---

## Timing guide (13 slides — now ~11-12 min, tight for a 12 min slot)

| Slide | Suggested time |
|---|---|
| 1 (title) | 20s |
| 2 (question) | 45s |
| 3 (why) | 60s |
| 4 (what we built) | 60s |
| 5 (objective 1) | 30s |
| 6 (before chart) | 60s |
| 7 (after chart) | 75s |
| 7a (diagnostics: what we measured) | 50s |
| 8 (objective 2) | 30s |
| 9 (memory wall) | 90s |
| 9a (memory scaling: what we measured + open problem) | 60s |
| 10 (why it matters) | 60s |
| 11 (closing) | 45s |

Adding the two new slides pushes the total to ~685s (~11.4 min) — still
inside 10-12 min, but with less room for Q&A spillover than before. If time
runs short in practice, slide 2 (45s) and slide 4 (60s) are the two safest
to compress on the fly — they're framing, not evidence.
| **Total** | **~9.5-10.5 min**, leaves room for questions inside a 12 min slot |
