This file is designed to guide you or another throughout the ideal working and build application of the event_data_workflow folder.

sys library helps manipulate different parts of python runtime environment.
this various from communicating using functions  within the system, via the CLI and especially during debugging. 

logging library helps  ecords events, errors, and diagnostic data during software execution, rather than using print to particularly show runtime results on teh terminal for momentarly. This would also also to resolve during software debugging. 

threading library helps the program to run multiple tasks concurrently within the same process. This is envoked given that a fucntion or a Class is defined in the file and addressed as argument into the function provided by threading library. It works in the backround. 

shutil library helps providing high-level operations for file and directory management.handles bulk, "human-scale" operations.

abc library (Abstract Base Class) is a blueprint for other classes. It defines a set of methods and properties that a subclass must create, while also preventing the abstract class itself from being directly instantiated. It's like a shady contract, the developer has to follow the requirmenets and cant create a subclass wihtout implementing those requried methods. 

collections library helps to write cleaner, faster, and more memory-efficient code. In simple words normal python provides the developer a list, tuple, dictionary. This library provides more depth with features. 

pathlib library provide an object-oriented interface for interacting with filesystem paths. Cross-Platform Compatibility.

typing library provides a way to explicitly declare the expected data types of variables, function arguments, and return values. So use of optional simple means the varible or function can be of any value and literal means fixed to a singular type. 

__future__ library provided by python to keep system python code operational reagrdless of newer verison. The agility. 

dataclasses library helps create cleaner, more maintainable classes for structured data.

then we use torch and tonic to throughly manage the daatset before explicit use, this is related to wen working with neuromorphic dataset. 

The project's hardware topology is fixed and singular: CPU always loads and caches recordings, GPU always trains. There is no separate GPU-only path, and VRAM is never used to store the dataset cache — only RAM and disk are cache storage targets. The cache controller's job is choosing where one dataset's cache lives among those two, not choosing between hardware configurations.

---

## BaseRecordingCache

BaseRecordingCache is the in-memory cache engine for raw neuromorphic recordings. It sits between the raw dataset on disk and the training loop — when a recording is requested, the cache serves it from RAM instead of re-reading from disk.

The eviction policy is plain FIFO: a single deque tracks insertion order, and the oldest entry is evicted once the cache is full. This pipeline's access pattern is a shuffled `DataLoader` — every recording is equally likely to recur each epoch, with no popularity skew for a more elaborate policy (e.g. S3-FIFO) to exploit, so plain FIFO gives the same practical hit rate with far less bookkeeping.

Two limits apply simultaneously: `max_recordings` (count cap) and `max_bytes` (RAM byte cap). Whichever is hit first triggers eviction. This prevents the cache from consuming unbounded memory regardless of how many or how large the recordings are.

The class is abstract — subclasses must implement `prepare_item()`, which defines what happens to a recording before it is stored.

### Subclass

**BoundedRecordingCache** — stores recordings in CPU RAM as-is. `prepare_item()` is a no-op (returns the raw recording unchanged). Used in hybrid mode as the hot layer on top of DiskCachedDataset.

### Bigger picture

`BaseRecordingCache` owns all the queuing, eviction, byte accounting, and thread safety. `BoundedRecordingCache` is CPU RAM only — there is no GPU-VRAM counterpart. `determine_dataset_strategy()` picks between `MemoryCachedDataset` (tonic), `DiskCachedDataset` (tonic), and `BoundedRecordingCache`-over-`DiskCachedDataset` (hybrid) based on live RAM/disk, so the rest of the pipeline never needs to know which one is running.

---

## Cache Engineering — Decision Flow

```
                    Raw dataset on disk
                           │
                    validate_first_sample()
                    (checks non-empty + measures first sample bytes)
                           │
                    estimate_dataset_memory_footprint()
                    (probes 10 random samples → extrapolates total GB)
                           │
                    SystemResourceMonitor.snapshot()
                    (live RAM, disk readings)
                           │
                    determine_dataset_strategy()
                           │
          ┌────────────────┼─────────────────────────────┐
          │                │                             │
   GPU pressure?     Enough RAM?               Enough disk?
   + disk exists     (threshold met)           (1.2× dataset)
          │                │                             │
          ▼                ▼                             ▼
   DiskCachedDataset  MemoryCachedDataset      DiskCachedDataset
   (avoid CUDA/RAM    (full dataset            (limited RAM
    competition)       in RAM)                  fallback)
                                    │
                            Large RAM (≥32GB)
                            + disk available?
                                    │
                                    ▼
                            BoundedRecordingCache      ← hybrid only
                            (RAM hot layer on top
                             of DiskCachedDataset)
                                    │
                            Nothing fits?
                                    │
                                    ▼
                            RuntimeError (system halt)
```

All modes flow into the DataLoader. The cache is always applied to raw recordings before temporal slicing, so N slices from the same recording share one cache entry.