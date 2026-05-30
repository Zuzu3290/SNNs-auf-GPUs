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

GPU_PHASE_CAPS is just a safety percentage for how much of the currently free GPU memory the cache is allowed to use in each phase. Warmup is when the GPU is starting to shape up, activations, CUDA kernels, and memory usage may not be stable yet.

---

## BaseS3FIFOCache

BaseS3FIFOCache is the in-memory cache engine for raw neuromorphic recordings. It sits between the raw dataset on disk and the training loop — when a recording is requested, the cache serves it from RAM instead of re-reading from disk.

The algorithm is S3-FIFO (Simple, Scalable, and Space-efficient FIFO), published at SOSP 2023. It uses three structures:

**Small queue (10% of cache)** — every new recording enters here first. If it gets accessed more than once before being evicted, it is promoted to the Main queue. If it is only accessed once, it is evicted and its index is added to the Ghost set (no data stored, just the index).

**Main queue (90% of cache)** — the stable working set. Uses a CLOCK-style second chance: when a recording is due for eviction, if it has been accessed recently its frequency counter is decremented and it is reinserted at the back of the queue. Only recordings with a frequency of zero are actually evicted.

**Ghost set** — a lightweight list of recently evicted indices (no data). If a recording that was previously evicted is requested again, it is admitted directly into Main instead of Small. This gives recurring recordings a fast path back into the hot layer.

Two limits apply simultaneously: `max_recordings` (count cap) and `max_bytes` (RAM byte cap). Whichever is hit first triggers eviction. This prevents the cache from consuming unbounded memory regardless of how many or how large the recordings are.

The class is abstract — subclasses must implement `prepare_item()`, which defines what happens to a recording before it is stored. This is where CPU vs GPU behaviour diverges.

### Subclasses

**BoundedRecordingCache** — stores recordings in CPU RAM as-is. `prepare_item()` is a no-op (returns the raw recording unchanged). Used in hybrid mode as the hot layer on top of DiskCachedDataset.

**GPURecordingCache** — stores recordings in CUDA VRAM. `prepare_item()` moves tensors to the GPU device. Used as a last resort when there is no disk and insufficient RAM, with a strictly computed VRAM budget so the cache never competes with model parameters or gradients.

### Bigger picture

`BaseS3FIFOCache` is the engine. It owns all the queuing, eviction, byte accounting, and thread safety. `BoundedRecordingCache` and `GPURecordingCache` are just two expressions of the same engine — one stores recordings in CPU RAM, the other in GPU VRAM. The only thing that differs between them is `prepare_item()`. `determine_dataset_strategy` decides which subclass gets instantiated based on available hardware, so the rest of the pipeline never needs to know which one is running.

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
                    (live RAM, disk, VRAM readings)
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
                            No disk + no RAM?
                            GPU free ≥ 0.5GB?
                                    │
                                    ▼
                            GPURecordingCache
                            (VRAM fallback,
                             lowest priority)
```

All modes flow into the DataLoader. The cache is always applied to raw recordings before temporal slicing, so N slices from the same recording share one cache entry.