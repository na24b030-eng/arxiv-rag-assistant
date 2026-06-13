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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0F1E;
    color: #E8EDF5;
}

.stApp {
    background-color: #0A0F1E;
}

/* ── Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Custom header */
.rag-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.25rem;
}

.rag-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin: 0;
}

.rag-title span {
    color: #00D4FF;
}

.rag-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #00D4FF;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.rag-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    color: #6B7FA3;
    margin: 0 0 2rem 0;
    font-weight: 400;
}

/* ── Search box */
.stTextInput > div > div > input {
    background-color: #111827 !important;
    border: 1px solid #1E2A3A !important;
    border-radius: 8px !important;
    color: #E8EDF5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.08) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #3D4F6E !important;
}

/* ── Search button */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0096FF 100%) !important;
    color: #0A0F1E !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
}

.stButton > button:disabled {
    background: #1E2A3A !important;
    color: #3D4F6E !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #1E2A3A !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #6B7FA3 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
    background: transparent !important;
}

/* ── Cards */
.answer-card {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    line-height: 1.7;
}

.paper-card {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}

.paper-card:hover {
    border-color: rgba(0, 212, 255, 0.3);
}

.paper-index {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #00D4FF;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.paper-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #E8EDF5;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

.paper-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6B7FA3;
    margin-bottom: 0.5rem;
}

.paper-meta a {
    color: #00D4FF;
    text-decoration: none;
}

.paper-meta a:hover {
    text-decoration: underline;
}

.paper-abstract {
    font-family: 'Inter', sans-serif;
    font-size: 0.825rem;
    color: #8B9DC3;
    line-height: 1.6;
    border-top: 1px solid #1E2A3A;
    padding-top: 0.75rem;
    margin-top: 0.75rem;
}

.rrf-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #6B7FA3;
    background: #0A0F1E;
    border: 1px solid #1E2A3A;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    float: right;
}

/* ── Contradiction badges */
.level-none {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #10B981;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
}

.level-low {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #60A5FA;
    background: rgba(96, 165, 250, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.25);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
}

.level-moderate {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #F59E0B;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
}

.level-high {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #EF4444;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
}

.contradiction-card {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 1.5rem;
}

/* ── Pipeline step indicators */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
}

.step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #1E2A3A;
    flex-shrink: 0;
}

.step-dot.active {
    background: #00D4FF;
    box-shadow: 0 0 6px rgba(0, 212, 255, 0.6);
}

.step-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #3D4F6E;
    letter-spacing: 0.05em;
}

.step-label.active {
    color: #6B7FA3;
}

/* ── Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0D1321 !important;
    border-right: 1px solid #1E2A3A !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: #0A0F1E !important;
}

.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: #3D4F6E;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.25rem 0 0.5rem 0;
}

.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1A2234;
}

.sidebar-stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #6B7FA3;
}

.sidebar-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00D4FF;
    font-weight: 500;
}

/* ── Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #1E2A3A;
    margin: 1.5rem 0;
}

/* ── Spinner override */
.stSpinner > div {
    border-top-color: #00D4FF !important;
}

/* ── Info box */
.stAlert {
    background: rgba(0, 212, 255, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #6B7FA3 !important;
}

/* ── Expander */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1E2A3A !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E8EDF5 !important;
}
</style>
""", unsafe_allow_html=True)

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

MODEL = "gemini-3.5-flash"

RAG_SYSTEM_PROMPT = """You are a precise ML research assistant. Answer using ONLY the provided context.

Formatting rules:
1. Cite every factual claim inline using [N] — e.g. "Transfer learning reuses features [1][3]."
2. If two sources CONFLICT, write: "Sources disagree: [A] says X while [B] says Y." Do NOT blend them.
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
# LOAD RESOURCES
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
# RETRIEVAL
# ============================================================

def tokenise(text):
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
            "metadata": {"arxiv_id": c["arxiv_id"], "title": c["title"], "categories": c["categories"]}
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
            "metadata": {"arxiv_id": c["arxiv_id"], "title": c["title"], "categories": c["categories"]}
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
    return reciprocal_rank_fusion(
        dense_retrieve(query, embed_model, embeddings, chunks),
        sparse_retrieve(query, bm25_index, chunks)
    )

# ============================================================
# GEMINI
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
    context = build_context_block(retrieved_chunks)
    prompt  = f"{RAG_SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{context}\n\n---\nQUESTION: {query}\n\nANSWER:"
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

def detect_contradictions(query, retrieved_chunks, client):
    context = build_context_block(retrieved_chunks)
    prompt  = f"{CONTRADICTION_SYSTEM_PROMPT}\n\nPapers retrieved for query: \"{query}\"\n\n{context}"
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

def get_contradiction_level(report_text):
    for level in ["HIGH", "MODERATE", "LOW", "NONE"]:
        if level in report_text.upper():
            return level
    return "UNKNOWN"

def get_level_html(level):
    css_class = f"level-{level.lower()}" if level in ["NONE","LOW","MODERATE","HIGH"] else "level-none"
    return f'<span class="{css_class}">{level}</span>'

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<p class="sidebar-section-label">Authentication</p>', unsafe_allow_html=True)
    api_key = st.text_input(
        "",
        type="password",
        placeholder="Gemini API key",
        label_visibility="collapsed"
    )
    st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com)")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-label">Knowledge Base</p>', unsafe_allow_html=True)

    stats = [
        ("Papers indexed", "5,000"),
        ("Embedding dim", "384"),
        ("Chunk strategy", "1:1 abstract"),
        ("Categories", "5 ML domains"),
    ]
    for label, val in stats:
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">{label}</span>
            <span class="sidebar-stat-value">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-label">Retrieval Pipeline</p>', unsafe_allow_html=True)

    pipeline_steps = [
        "Dense: MiniLM cosine",
        "Sparse: BM25Okapi",
        "Fusion: RRF (k=60)",
        f"Top-{TOP_K_FINAL} → Gemini",
    ]
    for step in pipeline_steps:
        st.markdown(f"""
        <div class="step-row">
            <div class="step-dot active"></div>
            <span class="step-label active">{step}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-label">Generation</p>', unsafe_allow_html=True)

    gen_steps = [
        "Model: Gemini 1.5 Flash",
        "Citations: inline [N]",
        "Contradiction scoring",
    ]
    for step in gen_steps:
        st.markdown(f"""
        <div class="step-row">
            <div class="step-dot active"></div>
            <span class="step-label active">{step}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="rag-header">
    <h1 class="rag-title">ArXiv <span>Research</span> Assistant</h1>
    <span class="rag-badge">RAG · v2</span>
</div>
<p class="rag-subtitle">
    Hybrid retrieval over 5,000 ML papers · Grounded answers with source citations · Cross-paper contradiction detection
</p>
""", unsafe_allow_html=True)

# ============================================================
# LOAD INDEXES
# ============================================================

embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ============================================================
# SEARCH INPUT
# ============================================================

col_input, col_btn = st.columns([5, 1])

with col_input:
    query = st.text_input(
        "",
        placeholder="Ask a research question — e.g. How does transfer learning work?  /  What is reinforcement learning?",
        label_visibility="collapsed"
    )

with col_btn:
    run_btn = st.button(
        "Search →",
        type="primary",
        disabled=not api_key or not query,
        use_container_width=True
    )

if not api_key:
    st.info("Add your Gemini API key in the sidebar to start querying the knowledge base.")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ============================================================
# EXAMPLE QUERIES
# ============================================================

if not query:
    st.markdown('<p class="sidebar-section-label" style="margin-bottom:0.75rem;">Try an example query</p>', unsafe_allow_html=True)
    ex_cols = st.columns(4)
    examples = [
        "What is transfer learning?",
        "How does reinforcement learning work?",
        "What are attention mechanisms?",
        "What is federated learning?",
    ]
    for col, example in zip(ex_cols, examples):
        with col:
            st.markdown(f"""
            <div style="
                background:#111827;
                border:1px solid #1E2A3A;
                border-radius:8px;
                padding:0.65rem 0.85rem;
                font-family:'Inter',sans-serif;
                font-size:0.78rem;
                color:#6B7FA3;
                cursor:pointer;
                line-height:1.4;
            ">{example}</div>
            """, unsafe_allow_html=True)

# ============================================================
# ON SEARCH
# ============================================================

if run_btn and api_key and query:

    client = genai.Client(api_key=api_key)

    with st.spinner("Scanning knowledge base via hybrid retrieval..."):
        retrieved = hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks)

    with st.spinner("Generating grounded answer with Gemini..."):
        answer = generate_grounded_answer(query, retrieved, client)

    with st.spinner("Running cross-paper contradiction analysis..."):
        contradiction_report = detect_contradictions(query, retrieved, client)

    contradiction_level = get_contradiction_level(contradiction_report)

    # ── Stats bar
    st.markdown(f"""
    <div style="
        display:flex;
        gap:2rem;
        padding:0.85rem 1.25rem;
        background:#111827;
        border:1px solid #1E2A3A;
        border-radius:10px;
        margin-bottom:1.5rem;
        align-items:center;
    ">
        <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D4F6E;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">Papers retrieved</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:700;color:#00D4FF;">{len(retrieved)}</div>
        </div>
        <div style="width:1px;background:#1E2A3A;height:2rem;"></div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D4F6E;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">Retrieval method</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.875rem;font-weight:600;color:#E8EDF5;">Dense + BM25 → RRF</div>
        </div>
        <div style="width:1px;background:#1E2A3A;height:2rem;"></div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D4F6E;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">Contradiction level</div>
            <div style="margin-top:0.1rem;">{get_level_html(contradiction_level)}</div>
        </div>
        <div style="width:1px;background:#1E2A3A;height:2rem;"></div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D4F6E;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">Generation model</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.875rem;font-weight:600;color:#E8EDF5;">Gemini 1.5 Flash</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs
    tab_answer, tab_sources, tab_contradiction = st.tabs([
        "📝  Grounded Answer",
        f"📄  Sources  ({len(retrieved)})",
        "⚖️  Contradiction Report"
    ])

    # ── Tab 1: Answer
    with tab_answer:
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="
            display:flex;
            align-items:center;
            gap:0.5rem;
            margin-top:0.75rem;
            padding:0.6rem 0.85rem;
            background:rgba(0,212,255,0.04);
            border:1px solid rgba(0,212,255,0.1);
            border-radius:8px;
        ">
            <span style="font-size:0.75rem;color:#3D4F6E;font-family:'Inter',sans-serif;">
                Every claim above is grounded in the retrieved papers. Citations like [1] refer to the Sources tab.
                Conflicting findings are surfaced explicitly rather than blended.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Sources
    with tab_sources:
        st.markdown(f"""
        <p style="font-family:'Inter',sans-serif;font-size:0.825rem;color:#6B7FA3;margin-bottom:1rem;">
            Retrieved via hybrid BM25 + dense semantic search, ranked by Reciprocal Rank Fusion score.
            Higher RRF score = stronger combined signal across both retrieval methods.
        </p>
        """, unsafe_allow_html=True)

        for i, paper in enumerate(retrieved, 1):
            arxiv_id = paper["metadata"]["arxiv_id"]
            title    = paper["metadata"]["title"]
            cats     = paper["metadata"]["categories"]
            rrf      = paper["rrf_score"]
            url      = f"https://arxiv.org/abs/{arxiv_id}"
            abstract = paper["document"][:350]

            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-index">
                    Source [{i}]
                    <span class="rrf-badge">RRF {rrf}</span>
                </div>
                <div class="paper-title">{title}</div>
                <div class="paper-meta">
                    <a href="{url}" target="_blank">arxiv.org/abs/{arxiv_id}</a>
                    &nbsp;·&nbsp; {cats}
                </div>
                <div class="paper-abstract">{abstract}...</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: Contradiction Report
    with tab_contradiction:
        st.markdown(f"""
        <p style="font-family:'Inter',sans-serif;font-size:0.825rem;color:#6B7FA3;margin-bottom:1rem;">
            Gemini analyses the {len(retrieved)} retrieved papers for agreements and conflicting findings.
            Most RAG systems silently blend contradictions — this system surfaces them explicitly.
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            display:flex;
            align-items:center;
            gap:1rem;
            padding:1rem 1.25rem;
            background:#111827;
            border:1px solid #1E2A3A;
            border-radius:10px;
            margin-bottom:1rem;
        ">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3D4F6E;text-transform:uppercase;letter-spacing:0.1em;">Contradiction level</span>
            {get_level_html(contradiction_level)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="contradiction-card">{contradiction_report}</div>', unsafe_allow_html=True)