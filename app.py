import re
import pickle
from collections import defaultdict

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from google import genai

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv Lens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #10121A;
    color: #C8D0E0;
}
.stApp { background: #10121A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1100px;
}

/* ── Hero ── */
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #F59E0B;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #F0F4FF;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    color: #F59E0B;
}
.hero-sub {
    font-size: 0.875rem;
    color: #5A677D;
    line-height: 1.6;
    max-width: 560px;
    margin: 0 0 2rem 0;
}

/* ── API banner ── */
.api-banner {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.75rem;
}
.api-banner-icon { font-size: 1.1rem; }
.api-banner-text {
    font-size: 0.8rem;
    color: #8A95A8;
    flex: 1;
}
.api-banner-text strong { color: #F59E0B; font-weight: 600; }

/* ── Search area ── */
.search-wrap {
    background: #181C28;
    border: 1.5px solid #232840;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.search-wrap:focus-within {
    border-color: #F59E0B;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.08);
}
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #F0F4FF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 0 !important;
    caret-color: #F59E0B;
}
.stTextInput > div > div > input::placeholder { color: #353D52 !important; }
.stTextInput > div { border: none !important; box-shadow: none !important; }
.stTextInput > label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: #F59E0B !important;
    color: #10121A !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #FBBF24 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(245,158,11,0.25) !important;
}
.stButton > button:disabled {
    background: #1E2535 !important;
    color: #374151 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Example pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }
.pill {
    font-family: 'Inter', sans-serif;
    font-size: 0.775rem;
    color: #6B7FA3;
    background: #181C28;
    border: 1px solid #232840;
    border-radius: 20px;
    padding: 0.35rem 0.85rem;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.pill:hover { border-color: #F59E0B; color: #F59E0B; }

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 0;
    background: #181C28;
    border: 1px solid #232840;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.stat-cell {
    flex: 1;
    padding: 0.85rem 1.1rem;
    border-right: 1px solid #232840;
}
.stat-cell:last-child { border-right: none; }
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #F0F4FF;
}
.stat-value.amber { color: #F59E0B; }
.stat-value.green { color: #10B981; }
.stat-value.blue  { color: #60A5FA; }
.stat-value.red   { color: #EF4444; }
.stat-value.gray  { color: #6B7FA3; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #232840 !important;
    gap: 0 !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #4B5563 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: #F59E0B !important;
    border-bottom: 2px solid #F59E0B !important;
}

/* ── Answer card ── */
.answer-body {
    background: #181C28;
    border: 1px solid #232840;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: #C8D0E0;
    margin-bottom: 0.75rem;
}
.answer-body h3 {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4B5563;
    margin: 1.25rem 0 0.5rem 0;
}
.grounding-note {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.7rem 1rem;
    background: rgba(245,158,11,0.04);
    border: 1px solid rgba(245,158,11,0.1);
    border-radius: 8px;
    font-size: 0.775rem;
    color: #4B5563;
    line-height: 1.5;
}

/* ── Source cards ── */
.source-card {
    background: #181C28;
    border: 1px solid #232840;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.65rem;
    transition: border-color 0.15s;
}
.source-card:hover { border-color: rgba(245,158,11,0.35); }
.source-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.source-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #F59E0B;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.15);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    flex-shrink: 0;
    letter-spacing: 0.06em;
}
.source-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    color: #E2E8F0;
    line-height: 1.4;
    flex: 1;
}
.rrf-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #374151;
    background: #10121A;
    border: 1px solid #1E2535;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    flex-shrink: 0;
}
.source-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #374151;
    margin-bottom: 0.65rem;
}
.source-meta a { color: #60A5FA; text-decoration: none; }
.source-meta a:hover { text-decoration: underline; }
.source-abstract {
    font-size: 0.8rem;
    color: #6B7FA3;
    line-height: 1.6;
    border-top: 1px solid #1E2535;
    padding-top: 0.65rem;
}

/* ── Contradiction ── */
.level-pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
}
.lvl-NONE     { color:#10B981; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); }
.lvl-LOW      { color:#60A5FA; background:rgba(96,165,250,0.08); border:1px solid rgba(96,165,250,0.2); }
.lvl-MODERATE { color:#F59E0B; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); }
.lvl-HIGH     { color:#EF4444; background:rgba(239,68,68,0.08);  border:1px solid rgba(239,68,68,0.2);  }
.lvl-UNKNOWN  { color:#6B7FA3; background:rgba(107,127,163,0.08); border:1px solid rgba(107,127,163,0.2); }

.contradiction-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: #181C28;
    border: 1px solid #232840;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.contradiction-header-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #374151;
}
.contradiction-body {
    background: #181C28;
    border: 1px solid #232840;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-size: 0.875rem;
    line-height: 1.75;
    color: #C8D0E0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D0F18 !important;
    border-right: 1px solid #1A1F2E !important;
}
.sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #F59E0B;
    margin-bottom: 0.15rem;
}
.sidebar-logo-sub {
    font-size: 0.7rem;
    color: #374151;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #2D3545;
    margin: 1.25rem 0 0.6rem 0;
}
.sidebar-kv {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid #141720;
}
.sidebar-kv-k { font-size: 0.775rem; color: #4B5563; }
.sidebar-kv-v {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #F59E0B;
    font-weight: 500;
}
.history-item {
    padding: 0.45rem 0.65rem;
    background: #141720;
    border: 1px solid #1A1F2E;
    border-radius: 6px;
    font-size: 0.775rem;
    color: #4B5563;
    margin-bottom: 0.4rem;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.history-item:hover { border-color: #F59E0B; color: #C8D0E0; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #F59E0B !important; }

/* ── Alert ── */
.stAlert {
    background: rgba(245,158,11,0.04) !important;
    border: 1px solid rgba(245,158,11,0.12) !important;
    border-radius: 8px !important;
    color: #5A677D !important;
    font-size: 0.825rem !important;
}

/* ── Divider ── */
hr { border-color: #1A1F2E !important; }
</style>
""", unsafe_allow_html=True)

# ── Configuration ────────────────────────────────────────────
EMBEDDINGS_PATH = "embeddings.npy"
CHUNKS_PATH     = "chunks.pkl"
BM25_PATH       = "bm25_index.pkl"
TOP_K_DENSE  = 10
TOP_K_SPARSE = 10
TOP_K_FINAL  = 5
RRF_K        = 60
MODEL        = "gemini-3.5-flash"

RAG_SYSTEM_PROMPT = """You are a precise ML research assistant. Answer using ONLY the provided context.
Rules:
1. Cite every claim inline as [N]. Example: "Transfer learning improves sample efficiency [1][3]."
2. If sources CONFLICT: "Sources disagree: [A] says X, [B] says Y." Never blend conflicting claims.
3. If not in context: "Not covered in the retrieved papers."
4. Max 300 words. End with ### Sources listing titles of [N] cited."""

CONTRADICTION_PROMPT = """You are a research analyst comparing ML papers.
Given these papers for the query "{query}":

{context}

Respond with:
1. CONTRADICTION LEVEL: NONE / LOW / MODERATE / HIGH
2. AGREEMENTS: 2-3 specific points of agreement across papers.
3. DISAGREEMENTS: Specific conflicting findings, or "None found."
Reference papers by [N]."""

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ── Load resources ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_embeddings():
    return np.load(EMBEDDINGS_PATH)

@st.cache_resource(show_spinner="Loading chunks...")
def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading BM25...")
def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

# ── Retrieval ─────────────────────────────────────────────────
def tokenise(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def dense_retrieve(query, embed_model, embeddings, chunks, top_k=TOP_K_DENSE):
    q_emb   = embed_model.encode([query])
    scores  = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{
        "chunk_id": chunks[i]["chunk_id"],
        "score":    float(scores[i]),
        "document": chunks[i]["chunk_text"],
        "metadata": {k: chunks[i][k] for k in ("arxiv_id","title","categories")}
    } for i in top_idx]

def sparse_retrieve(query, bm25_index, chunks, top_k=TOP_K_SPARSE):
    scores  = bm25_index.get_scores(tokenise(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{
        "chunk_id": chunks[i]["chunk_id"],
        "score":    float(scores[i]),
        "document": chunks[i]["chunk_text"],
        "metadata": {k: chunks[i][k] for k in ("arxiv_id","title","categories")}
    } for i in top_idx]

def rrf(dense, sparse, k=RRF_K, top_k=TOP_K_FINAL):
    scores, cmap = defaultdict(float), {}
    for rank, h in enumerate(dense, 1):
        scores[h["chunk_id"]] += 1/(k+rank); cmap[h["chunk_id"]] = h
    for rank, h in enumerate(sparse, 1):
        scores[h["chunk_id"]] += 1/(k+rank)
        if h["chunk_id"] not in cmap: cmap[h["chunk_id"]] = h
    result = []
    for cid, sc in sorted(scores.items(), key=lambda x: -x[1])[:top_k]:
        h = dict(cmap[cid]); h["rrf_score"] = round(sc, 6); result.append(h)
    return result

def hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks):
    return rrf(
        dense_retrieve(query, embed_model, embeddings, chunks),
        sparse_retrieve(query, bm25_index, chunks)
    )

# ── Gemini ────────────────────────────────────────────────────
def build_context(papers):
    return "\n\n".join(
        f"[{i}] Title: {p['metadata']['title']}\n    ArXiv: {p['metadata']['arxiv_id']}\n    Text: {p['document']}"
        for i, p in enumerate(papers, 1)
    )

def ask_gemini(prompt, client):
    try:
        return client.models.generate_content(model=MODEL, contents=prompt).text
    except Exception as e:
        return f"Gemini error: {e}"

def grounded_answer(query, papers, client):
    ctx = build_context(papers)
    return ask_gemini(
        f"{RAG_SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{ctx}\n\n---\nQUESTION: {query}\n\nANSWER:", client
    )

def contradiction_report(query, papers, client):
    ctx = build_context(papers)
    return ask_gemini(
        CONTRADICTION_PROMPT.format(query=query, context=ctx), client
    )

def get_level(text):
    for l in ["HIGH","MODERATE","LOW","NONE"]:
        if l in text.upper(): return l
    return "UNKNOWN"

def level_pill(level):
    return f'<span class="level-pill lvl-{level}">{level}</span>'

def level_color(level):
    return {"NONE":"green","LOW":"blue","MODERATE":"amber","HIGH":"red"}.get(level,"gray")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">ArXiv Lens</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-logo-sub">ML Research Assistant</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "", type="password",
        placeholder="Gemini API key",
        label_visibility="collapsed"
    )
    if api_key:
        st.markdown('<span style="font-size:0.72rem;color:#10B981;">✓ Key set — ready to search</span>', unsafe_allow_html=True)
    else:
        st.caption("Get one free at [aistudio.google.com](https://aistudio.google.com)")

    st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
    for k, v in [("Papers","5,000"),("Categories","cs.LG · cs.AI · cs.CL · cs.CV · stat.ML"),("Chunk strategy","1 abstract = 1 chunk"),("Embedding","MiniLM-L6-v2 · 384d")]:
        st.markdown(f'<div class="sidebar-kv"><span class="sidebar-kv-k">{k}</span><span class="sidebar-kv-v">{v}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Pipeline</div>', unsafe_allow_html=True)
    for step in ["Dense cosine search","BM25 keyword search","RRF fusion (k=60)","Gemini 3.5 Flash"]:
        st.markdown(f'<div style="font-size:0.775rem;color:#4B5563;padding:0.25rem 0;border-bottom:1px solid #141720;">→ {step}</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="sidebar-section">Recent queries</div>', unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-5:]):
            truncated = q[:42] + "…" if len(q) > 42 else q
            st.markdown(f'<div class="history-item" title="{q}">{truncated}</div>', unsafe_allow_html=True)

# ── Load indexes ──────────────────────────────────────────────
embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-eyebrow">5,000 ArXiv ML Papers · Hybrid Retrieval · Grounded by Gemini</div>
<h1 class="hero-title">Ask the research.<br><em>Get cited answers.</em></h1>
<p class="hero-sub">
    Type a question. ArXiv Lens retrieves the most relevant papers using both semantic and keyword search,
    then generates a grounded answer with inline citations — and tells you when papers disagree.
</p>
""", unsafe_allow_html=True)

# ── Search input ──────────────────────────────────────────────
st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
query = st.text_input(
    "",
    placeholder="e.g. How does attention mechanism work in transformers?",
    label_visibility="collapsed",
    key="query_input"
)
st.markdown('</div>', unsafe_allow_html=True)

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    search = st.button(
        "Search papers →",
        type="primary",
        disabled=not api_key or not query,
        use_container_width=True
    )
with col_hint:
    if not api_key:
        st.markdown('<span style="font-size:0.78rem;color:#374151;line-height:3rem;display:block;padding-top:0.6rem;">Add your Gemini API key in the sidebar to enable search.</span>', unsafe_allow_html=True)
    elif not query:
        st.markdown('<span style="font-size:0.78rem;color:#374151;line-height:3rem;display:block;padding-top:0.6rem;">Press Enter or click Search after typing your question.</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:0.78rem;color:#374151;line-height:3rem;display:block;padding-top:0.6rem;">Press Enter or click Search.</span>', unsafe_allow_html=True)

# ── Example queries ───────────────────────────────────────────
EXAMPLES = [
    "What is transfer learning?",
    "How does reinforcement learning work?",
    "What are attention mechanisms?",
    "What is federated learning?",
    "How does BERT work?",
]

if not query:
    st.markdown('<p style="font-size:0.72rem;color:#2D3545;font-family:JetBrains Mono,monospace;letter-spacing:0.1em;text-transform:uppercase;margin:1.25rem 0 0.6rem;">Try an example</p>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        with col:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state["query_input"] = ex
                st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Search logic ──────────────────────────────────────────────
# Trigger on button click OR Enter (query change from non-empty)
trigger = search or (query and query != st.session_state.last_query and api_key)

if trigger and api_key and query:
    st.session_state.last_query = query
    if query not in st.session_state.history:
        st.session_state.history.append(query)

    client = genai.Client(api_key=api_key)

    with st.spinner("Scanning 5,000 papers via hybrid retrieval..."):
        papers = hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks)

    with st.spinner("Generating grounded answer with Gemini 3.5 Flash..."):
        answer = grounded_answer(query, papers, client)

    with st.spinner("Running cross-paper contradiction analysis..."):
        report = contradiction_report(query, papers, client)

    level = get_level(report)
    lcolor = level_color(level)

    # ── Stats bar
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-cell">
            <div class="stat-label">Papers retrieved</div>
            <div class="stat-value amber">{len(papers)}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Retrieval method</div>
            <div class="stat-value">Dense + BM25 → RRF</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Contradiction level</div>
            <div class="stat-value">{level_pill(level)}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Top RRF score</div>
            <div class="stat-value amber">{papers[0]['rrf_score']}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Generation</div>
            <div class="stat-value">Gemini 3.5 Flash</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs
    tab_a, tab_s, tab_c = st.tabs([
        "📝  Answer",
        f"📄  Sources  ({len(papers)})",
        f"⚖️  Contradictions  ·  {level}"
    ])

    # Tab 1 — Answer
    with tab_a:
        st.markdown(f'<div class="answer-body">{answer}</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="grounding-note">
            <span>🔒</span>
            <span>Every claim is grounded in the retrieved papers above.
            Numbers like [1] refer to sources in the Sources tab.
            When papers conflict, disagreements are stated explicitly — not blended into false consensus.</span>
        </div>
        """, unsafe_allow_html=True)

    # Tab 2 — Sources
    with tab_s:
        st.markdown(f"""
        <p style="font-size:0.8rem;color:#4B5563;margin-bottom:1rem;">
            Ranked by Reciprocal Rank Fusion across dense semantic and BM25 keyword search.
            A higher RRF score means both retrieval methods agreed this paper is relevant.
        </p>
        """, unsafe_allow_html=True)

        for i, p in enumerate(papers, 1):
            aid   = p["metadata"]["arxiv_id"]
            title = p["metadata"]["title"]
            cats  = p["metadata"]["categories"]
            rrf_s = p["rrf_score"]
            url   = f"https://arxiv.org/abs/{aid}"
            abst  = p["document"][:320]

            st.markdown(f"""
            <div class="source-card">
                <div class="source-header">
                    <span class="source-num">SOURCE [{i}]</span>
                    <span class="source-title">{title}</span>
                    <span class="rrf-chip">RRF {rrf_s}</span>
                </div>
                <div class="source-meta">
                    <a href="{url}" target="_blank">arxiv.org/abs/{aid}</a> &nbsp;·&nbsp; {cats}
                </div>
                <div class="source-abstract">{abst}…</div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 3 — Contradictions
    with tab_c:
        st.markdown(f"""
        <p style="font-size:0.8rem;color:#4B5563;margin-bottom:1rem;">
            Gemini analyses the {len(papers)} retrieved papers for agreements and conflicts.
            Most RAG systems silently average contradictions into one confident answer.
            ArXiv Lens surfaces them explicitly.
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="contradiction-header">
            <span class="contradiction-header-label">Overall contradiction level</span>
            {level_pill(level)}
        </div>
        <div class="contradiction-body">{report}</div>
        """, unsafe_allow_html=True)