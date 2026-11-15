# MegaBake: elevator pitch

MegaBake is an open-source PyTorch compiler that turns any Pytorch model into a single persistent CUDA megakernel, with no model rewrite. You call `torch.compile(model, backend=megabake)` and keep the transformers model, cache and generation loop you already use. MegaBake takes the graph from there: it fuses the embedding, every layer, the KV-cache updates and the LM head into one cooperative kernel built from CuTe DSL bodies (Hopper TMA/WGMMA GEMMs, flash-decoding attention, fused norms and RoPE).

Megakernels remove kernel-launch gaps and keep the GPU streaming weights from start to finish. Today they are written by hand for one model at a time: vLLM's DeepSeek V4.1 path ships fused kernels such as `flash_mla_mega_attn` that are hard-coded to that model's shapes and to Blackwell. Every new model or GPU generation means writing them again; MegaBake's aim is to generate them automatically for any captured PyTorch workload. Because MegaBake is a compiler, each improvement lands on every model it supports at once, with no change to user code. It recovers tensor semantics and regions from post-grad FX, picks and tunes device bodies for the actual shapes, and plans their execution and synchronization inside one launch. It checks every result against eager PyTorch.

It has been tested on several model families, including Llama, Gemma, Qwen and DeepSeek, from 135M to 7B parameters. Each model compiles end to end into one single megakernel.

On performance, it already competes with the strongest baseline we know of, `torch.compile` max-autotune with CUDA graphs. On Hopper, decode is significantly faster on most of the models we benchmark, up to about 2x at long context, and it is on par at 7-8B scale.

And this is just the starting point. The next steps are clear and already on our roadmap: Hopper programmatic dependent launch, sharper GEMMs for narrow projections, FP8, and operator coverage that brings vision and audio models onto the same single-kernel path, and each one speeds up every supported model at once.

We are asking for support to get there: GPU access (full H100/H200/B200), review from CUTLASS/CuTe and transformers engineers, and a path to ship MegaBake as a backend that Hugging Face users can turn on with one line.

Code: https://github.com/megabake · Site: https://megabake.github.io
