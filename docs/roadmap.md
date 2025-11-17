# SNN-GPU-Event-Based Camera Diagram

```mermaid
graph LR
    %% Define nodes with bold and underlined text
    SNN["<b><u>Spiking Neural Network (SNN)</u></b>"]
    GPU["<b><u>Graphics Processing Unit (GPU)</u></b>"]
    EBC["<b><u>Event-Based Camera</u></b>"]

    %% Define arrows
    SNN --> GPU --> EBC
    EBC --> SNN

    %% Styles: white background, black border, thicker border
    style SNN fill:#FFFFFF,stroke:#000000,stroke-width:2px,color:#000000
    style GPU fill:#FFFFFF,stroke:#000000,stroke-width:2px,color:#000000
    style EBC fill:#FFFFFF,stroke:#000000,stroke-width:2px,color:#000000
```
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
