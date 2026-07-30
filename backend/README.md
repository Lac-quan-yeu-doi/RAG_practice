# Python Documentation RAG
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-7B68EE)
![SentenceTransformers](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-orange)
![BM25](https://img.shields.io/badge/Retrieval-BM25-blue)

The default project is a **Python Concurrency Assistant** using two logical reference documents:

1. **asyncio documentation**: selected `asyncio*.txt` files and the conceptual asyncio HOWTO.
2. **concurrent.futures documentation**: `library/concurrent.futures.txt`.

## 📁 Repository Structure

```text
rag-project/
├── benchmark/
│   └── example.jsonl
├── configs/
│   └── sources.json
├── data/
│   ├── raw/
│   └── vector_db/
├── scripts/
│   ├── ask.py
│   ├── build_index.py
│   ├── inspect_store.py
│   └── run_benchmark.py
├── src/rag_project/
│   ├── embeddings/
│   ├── evaluation/
│   ├── generation/
│   ├── loaders/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── storage/
│   ├── config.py
│   ├── factory.py
│   ├── pipeline.py
│   ├── schemas.py
│   └── text.py
├── tests/
├── .env
├── pyproject.toml
└── requirements.txt
```

Add `.env` file, for example:
```env
RAG_BASE_CONFIG=configs/sources.json

MAX_UPLOAD_MB=50
MAX_UNCOMPRESSED_MB=200
MAX_TEXT_FILES=2000
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8080

LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:1.7b
LLM_TIMEOUT_SECONDS=300
```



## 🚀 Installation
Run `setup.sh`

## 📚 Data documentations

Python 3.14 sources, download plain text to get `.txt` files: [Link](https://docs.python.org/3/download.html)


The documentation may remain elsewhere. Change `docs_root` in `configs/sources.json` to its actual location.

## ⚙️ Configure the Document Groups

`configs/sources.json` defines the corpus:

```json
{
  "documents": [
    {
      "document_id": "asyncio",
      "title": "Python asyncio documentation",
      "include": [
        "library/asyncio*.txt",
        "howto/a-conceptual-overview-of-asyncio.txt"
      ],
      "exclude": [
        "library/asyncio-api-index.txt",
        "library/asyncio-llapi-index.txt"
      ]
    },
    {
      "document_id": "concurrent_futures",
      "title": "Python concurrent.futures documentation",
      "include": [
        "library/concurrent.futures.txt"
      ],
      "exclude": []
    }
  ]
}
```
More sources can be added too.

## 🗄️ Build the Persistent Vector Database

```bash
python scripts/build_index.py --config configs/sources.json
```

This command preprocesses the text, creates embeddings, and stores in ChromaDB. The vector database is written to:

```text
data/vector_db/
```

Re-running the command rebuilds the configured collection.

## 🔍 Inspect Stored Chunks

```bash
python scripts/inspect_store.py   --config configs/sources.json   --limit 10
```

Filter by document group:

```bash
python scripts/inspect_store.py   --config configs/sources.json   --document-id asyncio   --limit 10
```

Use the printed chunk IDs when annotating the benchmark.

## 🔎 Run BM25 Retrieval

The default generator is extractive and does not require an LLM:

```bash
python scripts/ask.py   --config configs/sources.json   --retriever bm25   --generator extractive   --question "When should asyncio.gather() be used?"
```

Interactive mode:

```bash
python scripts/ask.py   --config configs/sources.json   --retriever bm25   --generator extractive   --interactive
```

BM25 loads the canonical chunks from ChromaDB and creates an in-memory sparse index. The vector database is therefore shared by all retrieval implementations.

## 🧠 Run Dense Retrieval

Dense retrieval is implemented separately in `retrieval/dense.py`:

```bash
python scripts/ask.py   --config configs/sources.json   --retriever dense   --generator extractive   --question "How can blocking work run without blocking the event loop?"
```

Switching retrievers does not require new preprocessing or a different database.

## 📊 Evaluation

### Generate evaulation question
```bash
python scripts/generate_benchmark.py \
  --config configs/sources.json \
  --output benchmark/generated.jsonl \
  --sections 20 \
  --questions-per-section 2 \
  --seed 42
```

### Evaluation

BM25:

```bash
python scripts/run_benchmark.py   --config configs/sources.json   --benchmark benchmark/generated.jsonl  --retriever bm25   --generator extractive   --top-k 5   --output outputs/bm25_results.json
```

Dense:

```bash
python scripts/run_benchmark.py   --config configs/sources.json   --benchmark benchmark/generated.jsonl  --retriever dense   --generator extractive   --top-k 5   --output outputs/dense_results.json
```
