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
