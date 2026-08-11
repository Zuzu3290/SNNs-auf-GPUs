# Dataset registry — history of changes

Background for why `dataset_registry.py` looks the way it does today. Kept separate so the
registry file itself stays short — code + one-line facts, not narrative.

## ASL-DVS removed (was slot "3")

Its official Dropbox link is disabled by the dataset owner (`shared_link_disabled`, confirmed
directly — not a bot-block). The README's alternate Google Drive link is a different, much
smaller artifact: 125 raw per-subject-per-letter `.aedat` recordings, not the 100,800 pre-sliced
`.mat` samples the registry entry pointed at. The source repo (NVS2Graph, github.com/PIX2NVS)
has no published raw-to-sliced conversion code, so self-slicing the raw files wouldn't be a
verified match to the paper's actual benchmark — real ASL-DVS sensor data, but not the same
distribution. No replacement found yet.

## CIFAR10-DVS tried, then rejected

Briefly used to fill ASL-DVS's slot (same host-fix pattern as DVS128 Gesture, verified
downloadable). Rejected: it's static-image classification (CIFAR-10 converted to DVS), not an
action-recognition dataset like ASL-DVS/DVS128 Gesture were. Removed from the registry.

## DHP19 (3D human pose) — investigated, not pursued

Candidate for a new *regression* entry (3D joint positions from Vicon mocap + 4 synchronized
DVS346 cameras). Real, accessible dataset — confirmed via a working ETH libdrive (Nextcloud)
link, WebDAV-listed: 17 subjects, ~145.5GB total, individual `.aedat` recordings ~100-166MB
each (a single-file test slice was well within reach). Not a dead end like ASL-DVS.

Decided against building it: the raw `.aedat` files are a multiplexed 4-camera format, and
correctly demultiplexing + aligning them to Vicon pose labels means translating the official
MATLAB toolbox (SensorsINI/DHP19 — `ImportAedatDataVersion1or2.m`,
`ExtractEventsToFramesAndMeanLabels.m`) into Python. Real, non-trivial work, not a quick
wrapper — deliberately not taken on. Not added to the registry.

## Regression loading generalized — `regression_datasets.py` removed

`DSECRaw` (one-off DSEC windowing class) replaced by `WindowedRecordingDataset`, a generic
wrapper now living in `dataset_registry.py` itself: given a recording source plus
`get_events`/`get_targets`/`get_windows` accessors, it flattens whole recordings into
`(event_window, target)` samples. DSEC's entry ("5") now builds `tonic.datasets.DSEC` directly
and wraps it inline via `_load_dsec()` — same behavior as `DSECRaw`, less code. Reusable for
DHP19 once it's accessible: parse its raw per-recording data however that dataset requires,
then hand `WindowedRecordingDataset` the three accessors for it.
