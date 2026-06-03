**An Even Easier Introduction to CUDA:**

https://learn.nvidia.com/courses/course-detail?course\_id=course-v1:DLI+T-AC-01+V1

**To Review**

https://developer.nvidia.com/content/life-triangle-nvidias-logical-pipeline
https://huggingface.co/blog/kernel-builder
https://www.ibm.com/think/topics/neural-processing-unit
https://safari.ethz.ch/architecture/fall2019/doku.php?id=schedule
https://interplayoflight.wordpress.com/2020/05/09/gpu-architecture-resources/
https://towardsdatascience.com/from-parallel-computing-principles-to-programming-for-cpu-and-gpu-architectures-dd06e1f30586/
https://medium.com/ai-insights-cobet/understanding-gpu-architecture-basics-and-key-concepts-40412432812b
https://www.linuxjournal.com/content/crafting-custom-linux-kernel-your-embedded-projects
https://huggingface.co/blog/kernel-builder
https://shivance.medium.com/your-very-own-cuda-kernel-3ed222be0d87
https://medium.com/@omkarpast/mastering-cuda-kernel-development-a-comprehensive-guide-1f3032666b94
https://docs.nvidia.com/cuda/cuda-c-programming-guide/
https://github.com/avishkarsaha/tutorials/blob/main/layernorm_cuda/layernorm-tutorial.md
https://avishkarsaha.com/2025/01/15/writing-custom-cuda-kernels-layernorm#:~:text=Writing%20custom%20CUDA%20kernels%20provides,Performance%20profiling%20and%20optimization%20strategies
https://www.youtube.com/watch?v=86FAWCzIe_4
https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

TPUs:
https://towardsdatascience.com/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a/



Alright the operators branch is focuse on completing on clear objective the scalabiity of the project performnce on using thre custom kernel. SO we need to address a basic kernel in crsc and then appliy a short conigurtaion file I guess in crossrefernce to C++/CUDA such that if we say to acclerate then we call the ideal folder acceleration that confines to operational optimized algotihrme ts to enhance overall use of the kernell, from energy supply, to memeory, to throughout optimization


Correct. They are fully independent subsystems with separate responsibilities:

src/compiler/ — operates at the graph level. It takes an SNN model, runs passes (device annotation, op rewrite, fusion), builds an IR, and decides what to execute and in what order. lif_kernel.h there declares the fused forward/backward interface the compiler lowers IR nodes into.

acceleration/ — operates at the hardware execution level. It doesn't know about computation graphs or IR. It only cares about how a single kernel launch runs: how much VRAM is free, what block size maximises SM occupancy, what the energy cost was.

src/crsc/ is the meeting point. engine.cu sits at the bottom of the compiler's lowering stack and at the top of the acceleration stack. The compiler hands it a tensor operation; it picks the right kernel path (lif_basic, lif_temporal, lif_warp_oriented) using KernelConfig from the acceleration layer.


src/compiler/   →  what to run, when, fused how
                         ↓
src/crsc/engine.cu  →  which kernel, which path
                         ↓
acceleration/   →  how to run it on the hardware
Neither the compiler nor the acceleration layer imports from the other. src/crsc/ is the only layer that touches both.