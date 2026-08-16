_This project has been created as part of the 42 curriculum by aarid._

# RAG-against-the-machine

A retrieval-augmented generation (RAG) project designed to answer questions about a repository or documentation corpus by grounding the answer in relevant source passages instead of relying only on a language model's memory.

## Description

RAG-against-the-machine builds a full RAG pipeline around a code/document dataset.
The system loads source files, segments them into meaningful chunks, indexes them
for retrieval, and then answers natural-language questions by combining the retrieved
evidence with a generation model. The goal is to help users ask precise questions
such as "Where is this feature implemented?" or "What is the intended behavior of
this component?" and receive answers backed by document evidence.

This project is especially relevant for codebases and technical documentation,
where a single model answer is often too generic unless it is anchored to the
actual source material. The stack combines lexical retrieval (BM25) with
optional semantic retrieval (FAISS + sentence embeddings) and can be used
either as a single-query search tool or as a batch evaluator over a dataset of questions.

## Instructions

### Requirements

- Python 3.12+
- `uv` package manager recommended

### Installation

**Recommended using virtual environment**

```bash
cd RAG-against-the-machine
uv venv
uv sync
```

**After dependencies installed, you can start interacting
with the system by the commands below:**

#### Indexing a source tree

The indexing step read a source tree and builds the RAG index
for later retrieval.

```bash
uv run python -m src.main index --max_chunk_size 2000 \
	--raw_path data/raw/vllm-0.10.1 \
	--processed_path data/processed/
```

###### Arguments:

- `--raw_path`: The path to the raw source files.
- `--processed_path`: The path to store processed files (index).
- `--max_chunk_size`: The maximum number of characters allowed in a chunk.

_Note: Project was build around vllm repo but i dont see what can
prevent it from working against other repos._

#### Running a single search

```bash
uv run python -m src.main search \
  --query "How to configure OpenAi server?" \
  --k 5 \
  --processed_path data/processed/ \
  --use_hybrid (optional)
  --use_semantic (optional)
```

###### Arguments:

- `--query`: The query to search for.
- `--k`: The number of top results to return.
- `--processed_path`: The path to read index from (index).
- `--use_hybrid`: Use hybrid retrieval (BM25 + semantic).
- `--use_semantic`: Use semantic retrieval (FAISS + sentence embeddings).

retun a list of chunks that are relevant to the query

#### Running a batch search

```bash
uv run python -m src.main search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results \
  --processed_path data/processed/ \
  --k 10
```

###### Arguments:

- `--dataset_path`: The path to the dataset file.
- `--save_directory`: The directory to save the search results.
- `--processed_path`: The path to read index from (index).
- `--k`: The number of top results to return for each question.
- `--use_hybrid`: Use hybrid retrieval (BM25 + semantic).
- `--use_semantic`: Use semantic retrieval (FAISS + sentence embeddings).

retun a list of lists of chunks that are relevant to the query

#### Running answer and answer_dataset

```bash
uv run python -m src.main answer \
  --query "How to configure OpenAi server?" \
  --k 5 \
  --generator_model Qwen3-0.6B \
  --cache_dir ... # optional to override default huggingface cache dir
  --processed_path data/processed/ \
  --
```

```bash
uv run python -m src.main answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

###### Arguments:

- `student_search_results_path`
- `save_directory`
- `generator_model_name = "Qwen/Qwen3-0.6B"`
- `cache_dir`
- `processed_path`

*Qwen3-0.6B has the following features*:

    Type: Causal Language Models
    Training Stage: Pretraining & Post-training
    Number of Parameters: 0.6B
    Number of Paramaters (Non-Embedding): 0.44B
    Number of Layers: 28
    Number of Attention Heads (GQA): 16 for Q and 8 for KV
    Context Length: 32,768


#### Evaluation

**You can check the evaluation results with a quick light metrix by
running the following command:**

```bash
uv run python -m src.main evaluate \

  --student_search_results_path ...,
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10
```

**Also you can run the full pipeline py a quick API***

### Running the API

```bash
sudo uv run fastapi dev src/api.py
```

```bash
uv run uvicorn src.api:app --reload
```

Then open the FastAPI docs at:

- http://127.0.0.1:8000/docs

Or via the command line:

```bash
curl "http://127.0.0.1:8000/search?query=What%20is%20the%20chunking%20strategy?&k=5&use_hybrid=true"
```

## Resources

### General references

- [A complete guide to RAG](https://www.mrlatte.net/en/research/2026/04/27/rag-complete-guide/)
- [Mastering RAG: a deep dive into embeddings](https://medium.com/@shravankoninti/mastering-rag-a-deep-dive-into-embeddings-b78782aa1259)
- [Understanding embeddings](https://www.youtube.com/watch?v=v6g8eo86T8A)
- [Chunking strategies for RAG systems](https://developer.ibm.com/articles/awb-enhancing-rag-performance-chunking-strategies/)
- [Grounded generation](https://zeroentropy.dev/concepts/grounded-generation/)
- [Hugging Face chat templating](https://huggingface.co/docs/transformers/en/chat_templating)
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/getting-started)

### Hugging Face and model references

- [Transformers documentation](https://huggingface.co/docs/transformers/models)
- [Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/2)
- [SentenceTransformers semantic search guide](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search)

### AI usage

AI tools supported the project in a few targeted ways:

- suggesting chunking and indexing tecknologies used often
- helping to draft prompt structure for grounded generation
- reviewing retrieval and evaluation logic for edge cases
- explaining and documenting retrieval concepts such as BM25, FAISS
- And of course write half of README

## System architecture

The architecture is composed of several cooperating modules:

- Document loading: reads project files from a raw data directory and normalizes them into structured document objects.
- Chunking: splits each file according to its type (Python, Markdown, Other files ignored) while preserving source offsets so that retrieved spans can be mapped back to the original file accurately.
- Indexing: builds a BM25 index and, optionally, a FAISS embedding index for semantic search.
- Retrieval: runs queries against the indexed chunks and ranks the most relevant results using keyword relevance and/or embedding similarity.
- Answer generation: passes the retrieved context to a causal language model (for this project, Qwen/Qwen3-0.6B is used as the default generator) with a strict context-grounded prompt.
- Evaluation: measures retrieval quality with recall@k metrics and IoU overlap against ground-truth file spans.

In short, the flow is:

1. Load raw documents
2. Chunk them into context windows
3. Build the retrieval index
4. Run the query
5. Retrieve the top-k chunks
6. Feed the relevant context into the generator
7. Evaluate answers or search results against benchmark data

## Chunking strategy

The chunker is designed to preserve semantic boundaries and character offsets.
At first i was trying to implement a custom AST(Abstract Syntax Tree) chunker but it was too slow,
also requires handling a lot of edge cases and it was not stable, so i falled back to a simple character-based chunker.

- It uses LangChain's `RecursiveCharacterTextSplitter` for structured splitting.
- File-type-aware logic is applied: Python files use Python-aware separators, while Markdown files use Markdown-aware separators.
- The default chunk size is 2000 characters with a 200-character overlap.
- Each chunk stores its original `start_index` and `end_index`, which allows precise reconstruction of the original document and robust evaluation against ground-truth spans.
- This minimizes information loss when a question requires context that spans several function or section boundaries.

Basically instead of working around a giant blob of text, the chunker splits it into smaller chunks that are easier to reason about and work with.

## Retrieval method

The project supports multiple retrieval modes:

- BM25 keyword retrieval is the default and is implemented with `bm25s`.
- Semantic retrieval can be enabled with a sentence-transformers embedding model and a FAISS index.
- Hybrid retrieval combines both sources using reciprocal rank fusion (RRF), which ranks a document by the sum of inverse-rank contributions coming from both retrieval systems.

The retrieval logic in `src/search.py` follows this pattern:

- tokenize the user query
- fetch the top-k candidates from BM25
- optionally fetch the top-k candidates from the embeddings index
- merge candidate sets by chunk ID
- sort the merged results by fused score
- return the best matches for generation

This hybrid flow is helpful for both keyword-heavy queries and semantically similar questions that do not share the exact same vocabulary as the source files.
In other simple words, the whole point of retrieval is getting the source chunk or parts of file that related to the quesiton keywords

## Performance analysis

The evaluation layer computes retrieval quality using recall@k, with the code measuring overlap between retrieved chunks and ground-truth source ranges. The project evaluates Recall@1, Recall@3, Recall@5, and Recall@10.

The key formula is:

- Intersection-over-union (IoU) is used to compare a retrieved span with a ground-truth span.
- A retrieved chunk counts as a hit if it overlaps the target range above a threshold (currently 0.05).
- Recall@k is then the proportion of ground-truth items found in the top-k results.

This is especially valuable in retrieval-heavy tasks because it measures not just whether the answer is correct, but whether the relevant evidence was actually retrieved. The repository's benchmark configuration expects strong performance thresholds, with documented examples using Recall@5 targets of at least 50% on code tasks and 80% on documentation tasks.

In practice, BM25 is often stronger on exact keyword matches, while the hybrid mode improves robustness on paraphrased or conceptually similar queries. The system is tuned to maximize evidence coverage, which matters more than purely generating a fluent answer without supporting context.

The other performance ascpects are mostly aobut the models used, even realying on the Qwen3-0.6B is smooth and enough, but its still require a proper device, and its still noticable as tasks like answer_dataset.

## Design decisions

Several design choices were made to keep the project simple, explainable, and effective:

- File-aware chunking: code and Markdown are treated differently to preserve structure.
- Character offsets: chunks keep positional metadata for precise mapping back to original files.
- Hybrid retrieval: lexical and semantic signals are fused to handle both exact and conceptual lookups.
- Query caching: repeated queries can reuse cached retrieval results and avoid unnecessary recomputation.
- Grounded generation: the generator is instructed to answer only from the provided context, forced just by a system prompt.

These choices keep the pipeline transparent: the user can inspect which chunks were used, why they were chosen, and how they are mapped to the original source.

## Challenges faced

The main difficulties encountered while building this project were not purely model-related but system-level:

- Chunk boundary issues: splitting long files without losing semantic continuity required careful overlap and file-type-specific segmentation.
- Index quality: keyword-only retrieval struggles with paraphrased questions, while embedding-only retrieval can miss rarer technical terms.
- Long-context limits: not all chunks can be fed into the generator without exceeding the model's context budget, so chunk size and retrieval count have to be balanced.
- Ground-truth evaluation complexity: source spans can overlap partially, so IoU-based matching is more reliable than naive exact-match checks.
- Runtime cost: embedding models and FAISS indexing are heavier than BM25 but improve semantic recall.

The project addresses these by using hybrid retrieval, chunk overlap, explicit span tracking, and an evaluation strategy that checks retrieval quality in a realistic way.

## Usage-Overview

As shown above, the usage is quite directive which solve the problem that RAG was invented to solve.
To make llm answer your questions based on the context you provide. so first we should provide it, to do so we we need to prepare the documents by spliting them and build
a clean chunks that respect the llms and human limits, and then take those chunks and build a table that is searchable with those chuks, isn't we want the model to answer
from our context? then we should make it able to do, and that the purpose of `index` part.
Next step is the real search, taking the query and process it, which allow you to search that index you already built and get the most relevant chunks, that what `search` for.
Then taking those relelvants sources you retrieved and simply pass them to the model to answer your question based on them and quite that what `answer` is for.
The rest is just for performance usage like processing a batch instead of one question at time, and same thing for search.
