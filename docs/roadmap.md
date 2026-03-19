# SNN-GPU-Event-Based Camera Diagram

```mermaid
graph TD
    %% ── Environment bands ──────────────────────────────────────────
    subgraph EBC["📷 Event-Based Camera"]
        M1["**Module 1 — Event Camera**\nDVS / DAVIS / Prophesee sensor\n→ (x, y, t, polarity) events"]
        M2["**Module 2 — Driver / SDK**\nlibcaer · Metavision SDK\nBAF · hot-pixel filter · ROI"]
        M3["**Module 3 — Event Buffer**\nRing buffer · time-window slicing\nvoxel grid (2, T, H, W)"]
    end

    subgraph GPU["⚡ GPU"]
        M4["**Module 4 — GPU Memory**\nPinned memory · async H2D\ndouble buffer · CUDA streams"]
        M5["**Module 5 — Custom CUDA / ARM Kernel**\nLIF threshold · NEON SIMD\nvoxel scatter · polarity merge"]
    end

    subgraph SNN["🧠 Spiking Neural Network"]
        M6["**Module 6 — SNN Simulation**\nConv-LIF encoder · snnTorch\nBPTT · surrogate gradients"]
        M7["**Module 7 — Output / Classification**\nRate decode · TTFS\nspike raster · class + confidence"]
    end

    %% ── Forward flow ───────────────────────────────────────────────
    M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7

    %% ── Feedback loop ──────────────────────────────────────────────
    M7 -.->|feedback · model updates · bias adaptation| M1

    %% ── Styles ─────────────────────────────────────────────────────
    style M1 fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style M2 fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style M3 fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style M4 fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    style M5 fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    style M6 fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style M7 fill:#EEEDFE,stroke:#534AB7,color:#3C3489
```

---
style M6 fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style M7 fill:#EEEDFE,stroke:#534AB7,color:#3C3489
## Installation

Follow these steps to install the necessary dependencies to run the application on your GPU related to this project.  

**[Installation.md](https://github.com/Zuzu3290/SNNs-auf-GPUs/blob/main/docs/installation.md)** 

# Project Mindmap

```mermaid
graph TD
    %% Main node
    SNN["<b><u>Spiking Neural Network (SNN)</u></b>"]

    %% First level
    SNN --> WHY["Why"]
    SNN --> WHAT["What"]
    SNN --> FEATURES["Features"]
    SNN --> OTHER_DESIGN["Other Design"]
    SNN --> MATH["Math"]
    SNN --> COMPUTING_APP["Computing Application"]

    %% Computing Application subnodes
    COMPUTING_APP --> CA_WHY["Why"]
    COMPUTING_APP --> CA_HOW["How"]
    COMPUTING_APP --> CA_WHERE["Where"]
    COMPUTING_APP --> CA_WHAT["What"]

    SNN --> HW_APP["Hardware Application"]
    HW_APP --> CUDA["CUDA"]
    CUDA --> GPUs["GPUs"]
    GPUs --> TPUs["TPUs"]
    TPUs --> NPUs["NPUs"]
    HW_APP --> CPUs["CPUs"]
    HW_APP --> NEURO_AI["Neuromorphic AI"]

    %% Neuromorphic AI subnodes
    NEURO_AI --> CAMERA["Camera"]
    CAMERA --> SENSOR["Sensor"]
    SENSOR --> INPUT_LAYER["Input Layer Computation"]
    %% Event-Based Camera
    EBC["Event-Based Camera (EBC)"]

    %% Connect sensors branch to EBC
    SENSOR --> EBC
    CAMERA --> EBC
    INPUT_LAYER --> EBC

    %% Style for all nodes
    style SNN fill:#FFFFFF,stroke:#000000,stroke-width:2px,color:#000000
    style WHY fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style WHAT fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style FEATURES fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style OTHER_DESIGN fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style MATH fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style COMPUTING_APP fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CA_WHY fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CA_HOW fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CA_WHERE fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CA_WHAT fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style HW_APP fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CUDA fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style GPUs fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style TPUs fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style NPUs fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CPUs fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style NEURO_AI fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style CAMERA fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style SENSOR fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style INPUT_LAYER fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style EBC fill:#FFFFFF,stroke:#000000,stroke-width:2px,color:#000000
```
## To-Dos
This section contains all the tasks and research items planned for the project.  

See the **[To-Dos Markdown file](https://github.com/Zuzu3290/SNNs-auf-GPUs/blob/main/docs/To-Do.md)** 


