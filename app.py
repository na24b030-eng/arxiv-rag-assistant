import re
import pickle
from collections import defaultdict

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from google import genai

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_PATH = "embeddings.npy"
CHUNKS_PATH     = "chunks.pkl"
BM25_PATH       = "bm25_index.pkl"

TOP_K_DENSE  = 10
TOP_K_SPARSE = 10
TOP_K_FINAL  = 5
RRF_K        = 60

MODEL = "gemini-2.0-flash"

RAG_SYSTEM_PROMPT = """You are a precise ML research assistant. Answer using ONLY the provided context.

Formatting rules:
1. Cite every factual claim inline using [N] — e.g. "Transfer learning reuses features [1][3]."
2. If two sources CONFLICT, write: "⚠ Sources disagree: [A] says X while [B] says Y." Do NOT blend them.
3. If the answer is not in the context, say: "Not covered in the retrieved papers."
4. Keep your answer under 300 words.
5. End with a ### Sources section listing each [N] you cited."""

CONTRADICTION_SYSTEM_PROMPT = """You are a research analyst comparing machine learning papers.
Analyse the provided papers and respond with:

1. CONTRADICTION LEVEL: one of NONE / LOW / MODERATE / HIGH

2. AGREEMENTS:
   List 2-3 specific points where the papers agree.

3. DISAGREEMENTS:
   List any specific points where the papers present different perspectives or
   conflicting findings. If none, write "None found."

Be precise. Reference papers by [N] number."""

# ============================================================
# LOAD RESOURCES — cached so they only load once per session
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading embeddings...")
def load_embeddings():
    return np.load(EMBEDDINGS_PATH)

@st.cache_resource(show_spinner="Loading chunks...")
def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

# ============================================================
# RETRIEVAL FUNCTIONS
# ============================================================

def tokenise(text: str) -> list:
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def dense_retrieve(query, embed_model, embeddings, chunks, top_k=TOP_K_DENSE):
    query_emb = embed_model.encode([query])
    scores    = cosine_similarity(query_emb, embeddings)[0]
    top_idx   = np.argsort(scores)[::-1][:top_k]
    hits = []
    for idx in top_idx:
        c = chunks[idx]
        hits.append({
            "chunk_id": c["chunk_id"],
            "score":    float(scores[idx]),
            "document": c["chunk_text"],
            "metadata": {
                "arxiv_id":   c["arxiv_id"],
                "title":      c["title"],
                "categories": c["categories"]
            }
        })
    return hits

def sparse_retrieve(query, bm25_index, chunks, top_k=TOP_K_SPARSE):
    tokens  = tokenise(query)
    scores  = bm25_index.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    hits = []
    for idx in top_idx:
        c = chunks[idx]
        hits.append({
            "chunk_id": c["chunk_id"],
            "score":    float(scores[idx]),
            "document": c["chunk_text"],
            "metadata": {
                "arxiv_id":   c["arxiv_id"],
                "title":      c["title"],
                "categories": c["categories"]
            }
        })
    return hits

def reciprocal_rank_fusion(dense_hits, sparse_hits, k=RRF_K, top_k=TOP_K_FINAL):
    rrf_scores = defaultdict(float)
    chunk_map  = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["chunk_id"]
        rrf_scores[cid] += 1.0 / (k + rank)
        chunk_map[cid]   = hit

    for rank, hit in enumerate(sparse_hits, start=1):
        cid = hit["chunk_id"]
        rrf_scores[cid] += 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = hit

    ranked  = sorted(rrf_scores.items(), key=lambda x: -x[1])
    results = []
    for cid, score in ranked[:top_k]:
        hit = dict(chunk_map[cid])
        hit["rrf_score"] = round(score, 6)
        results.append(hit)
    return results

def hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks):
    dense_hits  = dense_retrieve(query, embed_model, embeddings, chunks)
    sparse_hits = sparse_retrieve(query, bm25_index, chunks)
    return reciprocal_rank_fusion(dense_hits, sparse_hits)

# ============================================================
# GEMINI FUNCTIONS
# ============================================================

def build_context_block(retrieved_chunks):
    lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        lines.append(
            f"[{i}] Title : {chunk['metadata']['title']}\n"
            f"    ArXiv : {chunk['metadata']['arxiv_id']}\n"
            f"    Text  : {chunk['document']}"
        )
    return "\n\n".join(lines)

def generate_grounded_answer(query, retrieved_chunks, client):
    context  = build_context_block(retrieved_chunks)
    prompt   = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"---\nCONTEXT:\n{context}\n\n"
        f"---\nQUESTION: {query}\n\nANSWER:"
    )
    response = client.models.generate_content(
        model    = MODEL,
        contents = prompt
    )
    return response.text

def detect_contradictions(query, retrieved_chunks, client):
    context = build_context_block(retrieved_chunks)
    prompt  = (
        f"{CONTRADICTION_SYSTEM_PROMPT}\n\n"
        f"Papers retrieved for query: \"{query}\"\n\n"
        f"{context}"
    )
    response = client.models.generate_content(
        model    = MODEL,
        contents = prompt
    )
    return response.text

def get_contradiction_level(report_text):
    for level in ["HIGH", "MODERATE", "LOW", "NONE"]:
        if level in report_text.upper():
            return level
    return "UNKNOWN"

# ============================================================
# UI — HEADER
# ============================================================

st.title("🔬 ArXiv Research Assistant")
st.caption(
    "RAG pipeline · Dense + BM25 hybrid retrieval · "
    "Reciprocal Rank Fusion · Gemini grounded generation · "
    "Contradiction detection"
)
st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here"
    )
    st.caption(
        "Get a free key at [aistudio.google.com](https://aistudio.google.com)"
    )
    st.divider()
    st.markdown("**Knowledge base**")
    st.markdown("- 5,000 ArXiv ML abstracts")
    st.markdown("- Categories: cs.LG · cs.AI · cs.CL · cs.CV · stat.ML")
    st.divider()
    st.markdown("**Retrieval**")
    st.markdown("- Dense: MiniLM cosine similarity")
    st.markdown("- Sparse: BM25Okapi")
    st.markdown("- Fusion: RRF (k=60)")
    st.markdown(f"- Top {TOP_K_FINAL} papers sent to Gemini")
    st.divider()
    st.markdown("**Generation**")
    st.markdown("- Gemini 2.0 Flash")
    st.markdown("- Inline [N] citations")
    st.markdown("- Contradiction analysis")

# ============================================================
# LOAD INDEXES
# ============================================================

embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ============================================================
# QUERY INPUT
# ============================================================

query = st.text_input(
    "Ask a question about ML research",
    placeholder="e.g. What is transfer learning?  /  How does attention work?",
)

run_btn = st.button(
    "Search",
    type="primary",
    disabled=not api_key or not query
)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start.")

# ============================================================
# ON SEARCH
# ============================================================

if run_btn and api_key and query:

    client = genai.Client(api_key=api_key)

    # Step 1 — Hybrid Retrieval
    with st.spinner("Retrieving relevant papers..."):
        retrieved = hybrid_retrieve(
            query, embed_model, embeddings, bm25_index, chunks
        )

    # Step 2 — Grounded Answer
    with st.spinner("Generating grounded answer..."):
        answer = generate_grounded_answer(query, retrieved, client)

    # Step 3 — Contradiction Analysis
    with st.spinner("Analysing contradictions..."):
        contradiction_report = detect_contradictions(query, retrieved, client)

    contradiction_level = get_contradiction_level(contradiction_report)

    # ── Tabs
    tab_answer, tab_sources, tab_contradiction = st.tabs([
        "📝 Answer", "📄 Sources", "⚖️ Contradiction Report"
    ])

    with tab_answer:
        st.subheader("Grounded Answer")
        st.markdown(answer)

    with tab_sources:
        st.subheader(f"Retrieved Papers — top {len(retrieved)} by RRF")
        for i, paper in enumerate(retrieved, 1):
            arxiv_id = paper["metadata"]["arxiv_id"]
            title    = paper["metadata"]["title"]
            cats     = paper["metadata"]["categories"]
            rrf      = paper["rrf_score"]
            url      = f"https://arxiv.org/abs/{arxiv_id}"

            with st.expander(f"[{i}]  {title}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**ArXiv ID:** [{arxiv_id}]({url})")
                    st.markdown(f"**Categories:** {cats}")
                with col2:
                    st.metric("RRF score", rrf)
                st.caption(paper["document"][:400] + "...")

    with tab_contradiction:
        st.subheader("Contradiction Analysis")
        level_colors = {
            "NONE":     "green",
            "LOW":      "blue",
            "MODERATE": "orange",
            "HIGH":     "red",
            "UNKNOWN":  "gray"
        }
        color = level_colors.get(contradiction_level, "gray")
        st.markdown(
            f"**Contradiction level:** :{color}[**{contradiction_level}**]"
        )
        st.divider()
        st.markdown(contradiction_report)