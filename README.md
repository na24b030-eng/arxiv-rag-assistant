# ArXiv Research Assistant

A RAG system that answers questions about machine learning research using 5,000 ArXiv paper abstracts as its knowledge base. Every answer cites the specific papers it draws from, and a second analysis pass explicitly checks whether those papers agree or contradict each other — rather than silently blending conflicting sources into one confident-sounding response.

**[→ Live demo](YOUR_STREAMLIT_URL_HERE)** — bring your own Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

---

## The problem this solves

Standard LLM chat answers ML questions from memory. Two failure modes come with that:

1. **Hallucination** — the model confidently states facts, statistics, or paper titles that don't exist
2. **False consensus** — when the research literature actually disagrees, the model smooths it into one authoritative-sounding answer rather than telling you the field is unsettled

This system addresses both. Answers are generated strictly from retrieved paper text (not the model's memory), and when retrieved papers disagree with each other, the system tells you that explicitly instead of hiding it.

---

## How it works

### Two phases

**Offline** — runs once in a Kaggle notebook, produces three files:

```
ArXiv JSON snapshot (5M+ papers)
        │
        ▼
Filter: cs.LG · cs.AI · cs.CL · cs.CV · stat.ML
        │
        ▼
5,000 papers · strip LaTeX artifacts · 1 abstract = 1 chunk
        │
        ├──────────────────────────┬──────────────────────
        ▼                          ▼
all-MiniLM-L6-v2 embeddings   BM25Okapi keyword index
(5000 × 384 numpy matrix)      (tokenised chunk text)
        │                          │
   embeddings.npy            bm25_index.pkl        chunks.pkl
```

**Online** — runs per query, in Streamlit:

```
User query
        │
        ├──────────────────────┬──────────────────────
        ▼                       ▼
Dense retrieval              Sparse retrieval
cosine similarity            BM25Okapi
top 10                       top 10
        │                       │
        └──────────┬────────────┘
                   ▼
     Reciprocal Rank Fusion (k=60)
     → top 5 papers
                   │
        ┌──────────┴──────────┐
        ▼                      ▼
  Gemini: grounded         Gemini: contradiction
  answer + [N] citations    NONE / LOW / MODERATE / HIGH
        │                      │
        └──────────┬───────────┘
                   ▼
        3-tab Streamlit UI
```

### Why hybrid retrieval

Dense semantic search catches paraphrased queries but can miss exact technical terms. BM25 keyword search is the opposite — strong on exact matches, blind to synonyms. Neither is reliably better than the other across query types, so both run in parallel and their rankings are combined with Reciprocal Rank Fusion.

RRF specifically — rather than averaging raw scores — because cosine similarity scores (bounded 0 to 1) and BM25 scores (unbounded, corpus-dependent) aren't on comparable scales. Averaging them would let whichever method happens to produce larger numbers dominate. RRF only uses rank position, making it scale-invariant.

### Why two separate Gemini calls

The answer generation and the contradiction analysis use different system prompts with different jobs. Running them together in one prompt risks the model under-attending to one task. Two focused calls produce more reliable adherence to both the citation format and the contradiction taxonomy.

### Chunking: one abstract = one chunk

EDA on the dataset showed average abstract length ~150 words, maximum ~400 words. Both are within a single embedding's effective context. Further splitting would fragment already-compact, topically coherent units without improving retrieval.

---

## Project structure

```
arxiv-rag-assistant/
├── app.py                  Streamlit app — all logic in one file
├── requirements.txt
├── .python-version         Pinned to 3.11 (see note below)
├── .gitattributes          Git LFS rules
│
├── embeddings.npy           5000×384 float32 embedding matrix   [LFS]
├── chunks.pkl               Paper metadata aligned to embeddings [LFS]
└── bm25_index.pkl           Pre-fitted BM25Okapi index           [LFS]
```

---

## Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embedding model | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Dense retrieval | `numpy` + `sklearn.metrics.pairwise.cosine_similarity` |
| Sparse retrieval | `rank-bm25` — `BM25Okapi` |
| Rank fusion | Custom RRF implementation |
| LLM | Gemini `gemini-3.5-flash` via `google-genai` |
| Hosting | Streamlit Community Cloud |
| Large files | Git LFS |

---

## Running locally

### Prerequisites

- Python 3.11 specifically — see the note on Python version below
- Git LFS — [git-lfs.com](https://git-lfs.com)
- A Gemini API key — [aistudio.google.com](https://aistudio.google.com) (free)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-rag-assistant.git
cd arxiv-rag-assistant
git lfs pull                         # pulls the three large knowledge-base files
```

Create a Python 3.11 virtual environment:

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# Mac / Linux
python3.11 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, paste your API key in the sidebar, and search.

---

## Rebuilding the knowledge base

If you want to re-run the offline pipeline from scratch rather than using the pre-built files:

1. Add the Cornell ArXiv dataset to your Kaggle notebook (`Cornell-University/arxiv`)
2. Run the notebook top to bottom — it produces `embeddings.npy`, `chunks.pkl`, `bm25_index.pkl` in `/kaggle/working/`
3. Download and replace the three files in the repository root
4. Push via Git LFS

The notebook (`rag-llm-final.ipynb`) is included in the repository.

---

## Notes

**Python version** — the `tokenizers` package (a dependency of `sentence-transformers`) requires a Rust compiler via PyO3, which does not yet support Python 3.14. Python 3.11 is required. A `.python-version` file pins this for environments that respect it.

**ChromaDB** — an earlier version of this project used ChromaDB as the vector store. It was removed because its dependency chain (`protobuf` → `opentelemetry` → `tokenizers` via PyO3) failed to build on the deployment environment's Python 3.14. At 5,000 vectors, brute-force cosine similarity over a numpy matrix is functionally identical in latency to an HNSW index. ChromaDB would become worth revisiting past roughly 100K–1M vectors.

**Contradiction detection model** — a DeBERTa NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) would be the more rigorous approach to pairwise contradiction detection. A structured Gemini prompt is used here instead — a deliberate simplicity tradeoff with the limitation clearly named rather than hidden.

**No formal retrieval evaluation** — retrieval quality was assessed qualitatively across a fixed set of test queries. Properly measuring precision@k or recall@k requires a labeled relevance judgment dataset, which doesn't exist for this corpus yet.

---

## Known limitations

- Knowledge base covers abstracts only, not full paper text
- Corpus is a fixed ArXiv snapshot — newer papers require re-running the offline pipeline
- Contradiction level is extracted from Gemini's free-text response via string matching — structured JSON output would be more robust
- No automated test suite
- User queries are inserted into the Gemini prompt without prompt-injection sanitization

---

## License

MIT