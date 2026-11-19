# RAG

*This project has been created as part
of the 42 curriculum by satifi.*

## Description

RAG is a retrieval-augmented generation pipeline for answering questions about
a local collection of documentation and Python source code. It indexes files
from `data/raw`, retrieves the most relevant code or documentation chunks for a
question, and can optionally pass those chunks to a small language model to
produce a concise answer.

The project is intentionally split into three stages:

1. **Indexing** converts source files into chunks and stores the retrieval
   statistics in `data/processed`.
2. **Retrieval** ranks chunks with a BM25-style score and writes search results
   as JSON.
3. **Generation** supplies retrieved text as context to
   `Qwen/Qwen3-0.6B` and writes answers as JSON.

The `evaluate` command measures whether a relevant source interval was
retrieved. This makes retrieval quality measurable independently of the
language model's answer quality.

## System architecture

```text
data/raw (.py, .txt, .md)
          |
          v
   Indexer / chunker
          |
          +--> data/processed/chunks.json
          +--> data/processed/term_frequency.json
          +--> data/processed/inverse_document_frequency.json
          +--> data/processed/average_document_length.json
          |
          v
       Retrieval
   (combined source corpus)
          |
          v
   ranked source chunks
          |
          +--> search output JSON
          |
          v
   Model generator
          |
          v
       answer JSON
```

`src/index.py` discovers Python, Markdown, and text files, creates chunks, and
computes term frequency (TF), inverse document frequency (IDF), and average
document length (average DL) for the combined corpus. `src/search.py` loads
those caches and scores every query against that corpus. Commands in
`src/__main__.py` connect indexing, retrieval, generation, and evaluation.

## Chunking strategy

The indexer uses different segmentation strategies for docs and code:

- **Documentation (`.md` and `.txt`)** is split into character ranges of at
  most `max_chunk_size` (2,000 by default). Consecutive chunks overlap by
  approximately 300 characters so that information crossing a boundary
  remains retrievable.
- **Python (`.py`)** is parsed with Tree-sitter. Syntax-tree nodes are kept as
  chunks when they fit the configured limit. Small adjacent nodes are buffered
  until the minimum code chunk size (500 bytes) is reached, while nodes larger
  than the limit are recursively inspected. Character offsets are converted
  from Tree-sitter byte offsets to Python string offsets before they are stored.

Every chunk records its source path, first and last character indices, and
token count. The source filename is included when building the indexed token
sequence, which helps distinguish similarly named symbols in different files.

## Retrieval method

Queries are tokenized into lowercase identifiers using the regular expression
`[a-zA-Z_][a-zA-Z0-9_]*`; underscores are split and one-character tokens are
discarded. Unknown query terms are ignored.

For each query term, the retriever looks up the chunks containing that term and
adds a BM25-style contribution:

```text
score(term, chunk) =
    IDF(term) * TF(term, chunk) * (k1 + 1)
    ------------------------------------
    TF(term, chunk) + k1 * (1 - b + b * DL / averageDL)
```

The constants are `k1 = 0.8` and `b = 0.5`. IDF is computed as
`log2(number_of_chunks / number_of_chunks_containing_term)`. Scores from all
query terms are summed, then chunks are returned in descending score order.
The index is shared by Python, Markdown, and text files. During indexing, each
chunk also includes its source filename in the token sequence, which helps
distinguish similarly named symbols in different files.

## Performance analysis

Retrieval quality is measured with `src/recall.py` at `k = 1, 3, 5, 10`.
For each answered question, a result counts as a hit when:

1. the retrieved chunk belongs to the same file as the annotated source; and
2. the character intervals overlap by more than 5% intersection-over-union
   (IoU).

The evaluator reports `recall@1`, `recall@3`, `recall@5`, and `recall@10` as
fractions of evaluated questions. Run the evaluation commands in the
**Instructions** section to obtain the scores for a particular dataset and
index. The repository does not commit a single representative benchmark table,
because scores depend on the selected public/private corpus and the contents
of `data/raw`.

Indexing and dataset operations display progress bars. Indexing is performed
once and its caches are reused by subsequent searches. Retrieval is
cache-based and avoids recomputing TF/IDF for every query; generation is the
expensive optional stage because it loads the Qwen tokenizer/model and performs
autoregressive inference.

## Design decisions

- **Mixed code and documentation corpus:** code questions and prose questions
  have different chunking strategies, but share one combined index and its
  statistics.
- **BM25-style lexical retrieval:** it is deterministic, transparent, and
  does not require an embedding service or vector database.
- **Source offsets instead of copied text:** offsets keep the index compact and
  allow generation to read the current source text when context is assembled.
- **Tree-sitter for Python:** syntax-aware boundaries preserve functions,
  classes, and other meaningful code units better than fixed-width splitting.
- **JSON/Pydantic data contracts:** cached indexes, datasets, search results,
  and answers are validated against explicit models in `src/models.py`.
- **Optional generation:** retrieval and recall can be tested without
  downloading or running a language model.

## Challenges faced

- **Mixed content types:** fixed-size chunks are adequate for prose but can
  split Python constructs. The indexer therefore uses fixed overlapping windows
  for documentation and Tree-sitter nodes for Python.
- **Byte offsets versus character offsets:** Tree-sitter reports byte
  positions, while Python slicing uses character positions. A byte-to-character
  map is built before storing code chunk ranges.
- **Model context limits:** generation checks the assembled prompt against the
  model's maximum context length and reports an error instead of silently
  truncating context.

## Instructions

### Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Enough disk space for Python dependencies and the Hugging Face model cache
  when using `answer_dataset`

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

### Installation

```bash
make install
```

This runs `uv sync`. The Makefile uses the configured project environment and
cache locations. To override them, set `UV_PROJECT_ENVIRONMENT`,
`UV_CACHE_DIR`, or `HF_HOME` before running the command.

### Prepare and build the index

Place `.py`, `.md`, and `.txt` files under `data/raw`, then run:

```bash
make run ACTION="index"
```

The default maximum chunk size is 2,000. It can be changed through the Fire
command interface:

```bash
uv run python3 -m src index --max_chunk_size=1500
```

The generated files are written to `data/processed/`.

### Search one question

```bash
uv run python3 -m src search --query="How is the tokenizer initialized?" --k=5
```

The command prints JSON representations of the top-ranked source chunks. Use
`make run ACTION="search --query='your question' --k=5"` when the shell quoting
is appropriate for the question.

### Generate one answer

The `answer` command retrieves context and generates an answer for one
question:

```bash
uv run python3 -m src answer \
  --query="How is the tokenizer initialized?" \
  --k=5
```

The first generation command downloads `Qwen/Qwen3-0.6B` into the Hugging Face
cache. The model answers only from the retrieved context and reports when the
context does not support an answer.

### Search a dataset

For an unanswered-question dataset, run retrieval with the dataset's JSON path:

```bash
uv run python3 -m src search_dataset \
  --dataset_path=data/datasets/public/UnansweredQuestions/dataset_docs_public.json \
  --k=10 \
  --save_directory=data/output

uv run python3 -m src search_dataset \
  --dataset_path=data/datasets/public/UnansweredQuestions/dataset_code_public.json \
  --k=10 \
  --save_directory=data/output
```

The result is saved under `data/output/` using the input filename.

### Generate answers

After retrieval, generate answers from the saved search results:

```bash
uv run python3 -m src answer_dataset \
  --student_search_results_path=data/output/dataset_docs_public.json \
  --save_directory=data/output
```

This downloads `Qwen/Qwen3-0.6B` on first use and writes answers to the output
directory. The model is instructed to answer directly from the retrieved
context and to say that it cannot answer when the context is insufficient.

### Evaluate recall

Evaluate search results against the corresponding answered-question ground
truth:

```bash
uv run python3 -m src evaluate \
  --student_search_results_path=data/output/dataset_docs_public.json \
  --dataset_path=data/datasets/public/AnsweredQuestions/dataset_docs_public.json
```

The output contains the four metrics `recall@1`, `recall@3`, `recall@5`, and
`recall@10`.

### Quality checks and cleanup

```bash
make lint
make clean
```

`make lint` runs mypy and flake8. `make clean` removes Python caches.

## Resources

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) —
  foundational concepts and practical guidance for working with large language
  models.
- [A Crash Course on Building RAG Systems](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/) —
  an implementation-focused introduction to retrieval-augmented generation.
- [Understanding TF-IDF (Term Frequency-Inverse Document Frequency)](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/) —
  an introduction to the term-weighting concepts underlying lexical retrieval.
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) —
  an overview of the lexical ranking algorithm used for retrieval.
- [Understanding Okapi BM25 Document Ranking Algorithm](https://medium.com/@readwith_emma/understanding-okapi-bm25-document-ranking-algorithm-70d81adab001) —
  an additional explanation of BM25 scoring and document ranking.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/) —
  documentation for the transformer models.

### AI usage

AI was used during development to understand retrieval-augmented generation
systems and pipelines, research best practices, support benchmarking, and
explore chunking strategies. AI was also used to help write the project
documentation.
