# MegaBake North Star

MegaBake turns a captured PyTorch model into a planned, composable CUDA megakernel. PyTorch and Hugging Face remain the model, parameter, and runtime interface; MegaBake owns graph-region fusion, device-body selection, scheduling, and execution planning and everything else and is responsible for giving the final megakernel.

## Compilation path

1. TorchDynamo captures the model invocation as an FX graph and supplies its specialization guards.
2. TorchInductor runs its post-grad FX transformations, including normalization and functionalization.
3. At a version-pinned handoff before Inductor lowering and scheduling, MegaBake receives the still-FX graph and its shape, layout, and guard metadata. This handoff is an owned integration with Inductor internals, not an assumed stable public API.
4. MegaBake performs FX-aware pattern matching and fusion, dividing the graph into regions with explicit inputs, outputs, dependencies, state effects, layouts, and numerical requirements.
5. Each supported region maps to a device-callable CuTe DSL body. MegaBake plans compatible tiling, CTA work assignment, storage lifetimes, synchronization, and stage ordering, then composes the selected bodies into one persistent CUDA kernel where legal.

## Composition rule

An FX region describes tensor work; it is not itself a GPU kernel. A region can join the megakernel only when its implementation exposes device-side work that can execute inside the enclosing kernel and satisfies the enclosing plan's data ownership, synchronization, and resource constraints. Separately launched kernel wrappers do not become composable merely because they were generated from CuTe DSL.

## Correctness and performance

The original captured graph remains the semantic reference. MegaBake preserves graph guards, outputs, mutation and state updates, aliasing, and numerical behavior through every fusion. The selected megakernel must pass complete-step correctness checks and be measured end to end against an equivalent `torch.compile` baseline for the same model, inputs, precision, and GPU. Support and performance are reported for explicit workload and GPU targets; they are not inferred for unmeasured shapes or architectures.

The north star is a small, reliable compiler path from normalized FX regions to a measured CuTe megakernel—not a requirement that every FX operation or every model fit one.
