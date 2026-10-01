.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) are currently the most performant execution method of vLLM on Intel Gaudi. When HPU Graphs are enabled, execution graphs will be traced (recorded) ahead of time (after performing warmup), to be later replayed during inference, significantly reducing host overheads. Recording can take large amounts of memory, which needs to be taken into account when allocating KV cache. Enabling HPU Graphs will impact the number of available KV cache blocks, but vLLM provides user-configurable variables to control memory management.

When HPU Graphs are being used, they share the common memory pool ("usable memory") as KV cache, determined by `gpu_memory_utilization` flag (`0.9` by default).
Before KV cache gets allocated, model weights are loaded onto the device, and a forward pass of the model is executed on dummy data, to estimate memory usage.
Only after that, `gpu_memory_utilization` flag is utilized - at its default value, will mark 90% of free device memory at that point as usable.
Next, KV cache gets allocated, model is warmed up, and HPU Graphs are captured.
Environment variable `VLLM_GRAPH_RESERVED_MEM` defines the ratio of memory reserved for HPU Graphs capture.
With its default value (`VLLM_GRAPH_RESERVED_MEM=0.1`), 10% of usable memory will be reserved for graph capture (later referred to as "usable graph memory"), and the remaining 90% will be utilized for KV cache.
Environment variable `VLLM_GRAPH_PROMPT_RATIO` determines the ratio of usable graph memory reserved for prefill and decode graphs. By default (`VLLM_GRAPH_PROMPT_RATIO=0.3`), both stages have equal memory constraints.
Lower value corresponds to less usable graph memory reserved for prefill stage, e.g. `VLLM_GRAPH_PROMPT_RATIO=0.2` will reserve 20% of usable graph memory for prefill graphs, and 80% of usable graph memory for decode graphs.

!!! note
    `gpu_memory_utilization` does not correspond to the absolute memory usage