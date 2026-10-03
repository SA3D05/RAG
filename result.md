un the benchmarking script

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

If successful, you will see the following output

```text
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total Token throughput (tok/s):          619.85
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```

### Custom Dataset

If the dataset you want to benchmark is not supported yet in vLLM, even then you can benchmark on it using `CustomDataset`. Your data needs to be in `.jsonl` format and needs to have "prompt" field per entry, e.g., data.jsonl

```json
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

```bash
# start server
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```bash
# run benchmarkings for vLLM contributors

- [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [Dynamic speculative decoding](gh-issue:4565) }'
    ```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693570,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 0.001094818115234375
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

You can pass a list to both `text_1` and `text_2`, forming multiple sentence pairs
where each pair is built from a string in `text_1` and the corresponding string in `text_2` (similar to `zip()`).
The total number of pairs is `len(text_2)`.

??? console "Request"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "encoding_format": "float",
      "text_1": [
        "What is the capital of Brazil?",
        "What is the capital of France?"
      ],
      "text_2": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

#### Multi-modal inputs

You can pass multi-modal inputs to scoring models by passing `content` including a list of multi-modal input (image, etc.) in the request. Refer to the examples below for illustration.

=== "JinaVL-Reranker"

    To serve the model:

    ```bash
    vllm serve jinaai/jina-reranker-m0
    ```

    Since the request schema is not defined by OpenAI client, we post a request to the3bbQ)
- [Inference Perf](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf)
- <gh-issue:5041> and <gh-pr:12726>.
  
This is a non-trivial topic. Consider this comment from Rob:

> I think this metric should focus on trying to estimate what the max
> concurrency that will cause the average request length > queries per
> second ... since this is really what will "saturate" the server.

A clear goal is that we should expose the metrics required to detect
this saturation point, so administrators can implement auto-scaling
rules based on those. However, in order to do so, we need to have a
clear view on how an administrator (and automated monitoring system)
should judge an instance as approaching saturation:

> To identify, what is the saturation point for model server compute
> (the inflection point where we cannot get more throughput with a
> higher request rate, but start to incur additional latency) so we
> can autoscale effectively?

### Metric Naming

Our approach to naming metrics probably deserves to be revisited:

1. The use of colons in metric names seems contrary to
   ["colons are reserved for user defined recording rules"](https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels).
2. Most of our metrics follow the convention of ending with units, but
   not all do.
3. Some of our metric names end with `_total`:

    If there is a suffix of `_total` on the metric name, it will be removed. When
    exposing the time series for counter, a `_total` suffix will be added. This is
    for compatibility between OpenMetrics and the Prometheus text format, as OpenMetrics
    requires the `_total` suffix.

### Adding More Metrics

There is no shortage of ideas for new metrics:

- Examples from other projects like
  [TGI](https://github.com/IBM/text-generation-inference?tab=readme-ov-file#metrics)
- Proposals arising from specific use cases, like the Kubernetes
  auto-scaling topic above
- Proposals that might arise out# LangChain

vLLM is also available via [LangChain](https://github.com/langchain-ai/langchain) .

To install LangChain, run

```bash
pip install langchain langchain_community -q
```

To run inference on a single or multiple GPUs, use `VLLM` class from `langchain`.

??? code

    ```python
    from langchain_community.llms import VLLM

    llm = VLLM(model="mosaicml/mpt-7b",
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=128,
            top_k=10,
            top_p=0.95,
            temperature=0.8,
            # tensor_parallel_size=... # for distributed inference
    )

    print(llm("What is the capital of France ?"))
    ```

Please refer to this [Tutorial](https://python.langchain.com/docs/integrations/llms/vllm) for more details. CPU supports quantizations:
    - AWQ (x86 only)
    - GPTQ (x86 only)
    - compressed-tensor INT8 W8A8 (x86, s390x)

### (x86 only) What is the purpose of `VLLM_CPU_MOE_PREPACK` and `VLLM_CPU_SGL_KERNEL`?

- Both of them requires `amx` CPU flag.
    - `VLLM_CPU_MOE_PREPACK` can provides better performance for MoE models
    - `VLLM_CPU_SGL_KERNEL` can provides better performance for MoE models and small-batch scenarios.`vllm >= 0.7.0`.

### Step 1: Create your batch file

Add score requests to your batch file. The following is an example:

```text
{"custom_id": "request-1", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is the capital of France?", "text_2": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is the capital of France?", "text_2": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
```

You can mix chat completion, embedding, and score requests in the batch file, as long as the model you are using supports them all (note that all requests must use the same model).

### Step 2: Run the batch

You can run the batch using the same command as in earlier examples.

### Step 3: Check your results

You can check your results by running `cat results.jsonl`

```bash
cat results.jsonl
{"id":"vllm-f87c5c4539184f618e555744a2965987","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-806ab64512e44071b37d3f7ccd291413","body":{"id":"score-4ee45236897b4d29907d49b01298cdb1","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.0010900497436523438},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
{"id":"vllm-41990c51a26d4fac8419077f12871099","custom_id":"request-2","response":{"status_code":200,"request_id":"vllm-batch-73ce66379026482699f81974e14e1e99","body":{"id":"score-13f2ffe6ba40460fbf9f7f00ad667d75","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.001094818115234375},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_detaing op, the schema must be updated
  to reflect the changes.
- If a new custom type is needed, see the following document:
  [Custom Class Support in PT2](https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA).

### Notes for Large Changes

Please keep the changes as concise as possible. For major architectural changes
(>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue
(RFC) discussing the technical design and justification. Otherwise, we will tag
it with `rfc-required` and might not go through the PR.

### What to Expect for the Reviews

The goal of the vLLM team is to be a *transparent reviewing machine*. We would
like to make the review process transparent and efficient and make sure no
contributor feels confused or frustrated. However, the vLLM team is small, so we
need to prioritize some PRs over others. Here is what you can expect from the
review process:

- After the PR is submitted, the PR will be assigned to a reviewer. Every
  reviewer will pick up the PRs based on their expertise and availability.
- After the PR is assigned, the reviewer will provide status updates every 2-3
  days. If the PR is not reviewed within 7 days, please feel free to ping the
  reviewer or the vLLM team.
- After the review, the reviewer will put an `action-required` label on the PR
  if there are changes required. The contributor should address the comments and
  ping the reviewer to re-review the PR.
- Please respond to all comments within a reasonable time frame. If a comment
  isn't clear or you disagree with a suggestion, feel free to ask for
  clarification or discuss the suggestion.
- Note that not all CI checks will be executed due to limited computational
  resources. The reviewer will add `ready` label to the PR when the PR is
  ready to merge or a full CI run is needed.

## Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM.
All of your contributionspply a cross-encoder model or an embedding model to predict scores for sentence or multimodal pairs. When using an embedding model the score corresponds to the cosine similarity between each embedding pair.
Usually, the score for a sentence pair refers to the similarity between two sentences, on a scale of 0 to 1.

You can find the documentation for cross encoder models at [sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html).

Code example: <gh-file:examples/online_serving/openai_cross_encoder_score.py>

#### Single inference

You can pass a string to both `text_1` and `text_2`, forming a single sentence pair.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "text_1": "What is the capital of France?",
  "text_2": "The capital of France is Paris."
}'
```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

#### Batch inference

You can pass a string to `text_1` and a list to `text_2`, forming multiple sentence pairs
where each pair is built from `text_1` and a string in `text_2`.
The total number of pairs is `len(text_2)`.

??? console "Request"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "text_1": "What is the capital of France?",
      "text_2": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "objecte 1"
    We only cache full blocks.

!!! note "Note 2"
    The above hash key structure is not 100% collision free. Theoretically it’s still possible for the different prefix tokens to have the same hash value. To avoid any hash collisions **in a multi-tenant setup, we advise to use SHA256** as hash function instead of the default builtin hash.
    SHA256 is supported since vLLM v0.8.3 and must be enabled with a command line argument. It comes with a performance impact of about 100-200ns per token (~6ms for 50k tokens of context).

**A hashing example with multi-modality inputs**  
In this example, we illustrate how prefix caching works with multi-modality inputs (e.g., images). Assuming we have a request with the following messages:

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

It will become the following prompt:

```text
Prompt:
    <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

As we can see, after the tokenization, the `[IMG]` will be replaced by a sequence of placeholder tokens, and these placeholders will be replaced by image embeddings during prefill. The challenge for prefix caching to support this case is we need to differentiate images from the placeholders. To address this problem, we encode the image hash generated by the frontend image processor. For example, the hash of the blocks in the above prompt would be (assuming block size 16, and we have 41 placeholder tokens):

```text
Block 0
    Parent hash: None
    Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    Extra hash: <image hash>
Block 1
    Parent hash: Block 0 hash
    Token IDs: <p>, ..., <p>
    Extra 