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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F5F0;
    color: #1A1814;
}
.stApp { background-color: #F7F5F0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 4rem;
    max-width: 1080px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E8E4DC !important;
}
section[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem !important;
}
.sidebar-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 18px;
    color: #1A1814;
    margin-bottom: 2px;
}
.sidebar-brand-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8C8680;
    margin-bottom: 1.5rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #E8E4DC;
    margin: 1.25rem 0;
}
.sidebar-section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8C8680;
    margin: 0 0 0.6rem 0;
}
.sidebar-kv {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.4rem 0;
    border-bottom: 1px solid #E8E4DC;
    gap: 0.5rem;
}
.sidebar-kv-k {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #8C8680;
    flex-shrink: 0;
}
.sidebar-kv-v {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #1A1814;
    font-weight: 500;
    text-align: right;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #E8E4DC;
}
.step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #C8B89A;
    flex-shrink: 0;
}
.step-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #1A1814;
}
.history-pill {
    display: block;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #8C8680;
    background: #EDEAE4;
    border: 1px solid #DDD9D2;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.4rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: default;
}
.key-status-ok {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #4A7C59;
    background: #F0F7F0;
    border: 1px solid #C3DCC9;
    border-radius: 6px;
    padding: 0.35rem 0.65rem;
    margin-top: 0.4rem;
    display: inline-block;
}

/* ── Sidebar text input ── */
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #F7F5F0 !important;
    border: 1px solid #E8E4DC !important;
    border-radius: 8px !important;
    color: #1A1814 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    padding: 0.5rem 0.75rem !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
    color: #B8B0A4 !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: #C8B89A !important;
    box-shadow: 0 0 0 3px rgba(200,184,154,0.15) !important;
}
section[data-testid="stSidebar"] .stTextInput > label { display: none !important; }

/* ── Hero ── */
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #C8B89A;
    margin-bottom: 0.75rem;
}
.hero-heading {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 5vw, 58px);
    font-weight: 400;
    color: #1A1814;
    line-height: 1.12;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.01em;
}
.hero-heading em {
    font-style: italic;
    color: #C8B89A;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 300;
    color: #8C8680;
    line-height: 1.65;
    max-width: 520px;
    margin: 0.75rem 0 2.25rem 0;
}

/* ── Search box ── */
.search-outer {
    background: #FFFFFF;
    border: 1.5px solid #E8E4DC;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.search-outer:focus-within {
    border-color: #C8B89A;
    box-shadow: 0 0 0 4px rgba(200,184,154,0.12);
}
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #1A1814 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    padding: 0 !important;
    caret-color: #C8B89A;
}
.stTextInput > div > div > input::placeholder { color: #B8B0A4 !important; }
.stTextInput > div { border: none !important; box-shadow: none !important; }
.stTextInput > label { display: none !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #1A1814 !important;
    color: #F7F5F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2D2922 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(26,24,20,0.18) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: #E8E4DC !important;
    color: #B8B0A4 !important;
}

/* ── Secondary / example buttons ── */
.stButton > button[kind="secondary"] {
    background: #EDEAE4 !important;
    color: #1A1814 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    border: 1px solid #DDD9D2 !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    width: 100% !important;
    transition: all 0.15s !important;
    text-align: left !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #FFFFFF !important;
    border-color: #C8B89A !important;
    color: #1A1814 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E8E4DC !important;
    gap: 0 !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    color: #8C8680 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: #1A1814 !important;
    border-bottom: 2px solid #C8B89A !important;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.75rem;
}
.stat-cell {
    flex: 1;
    padding: 0.85rem 1.1rem;
    border-right: 1px solid #E8E4DC;
}
.stat-cell:last-child { border-right: none; }
.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8C8680;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1A1814;
    line-height: 1.2;
}
.stat-value.gold { color: #C8B89A; }
.stat-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #8C8680;
    margin-top: 0.1rem;
}

/* ── Level pills ── */
.lvl-pill {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
}
.lvl-NONE     { color:#3D6B4F; background:#EBF5EE; border:1px solid #C3DCC9; }
.lvl-LOW      { color:#2E5FA3; background:#EBF0FA; border:1px solid #BDD0F0; }
.lvl-MODERATE { color:#8A6020; background:#FDF3E3; border:1px solid #E8D5A8; }
.lvl-HIGH     { color:#8B2525; background:#FDF0F0; border:1px solid #E8BEBE; }
.lvl-UNKNOWN  { color:#8C8680; background:#EDEAE4; border:1px solid #DDD9D2; }

/* ── Answer card ── */
.answer-card {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 300;
    line-height: 1.8;
    color: #1A1814;
    margin-bottom: 1rem;
}
.answer-card strong { font-weight: 600; }
.grounding-note {
    background: #F7F5F0;
    border: 1px solid #E8E4DC;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #8C8680;
    line-height: 1.5;
}

/* ── Source cards ── */
.source-card {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.source-card:hover {
    border-color: #C8B89A;
    box-shadow: 0 2px 12px rgba(200,184,154,0.12);
}
.source-top {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    margin-bottom: 0.5rem;
}
.source-num {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #C8B89A;
    background: #FAF8F5;
    border: 1px solid #E8E4DC;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 2px;
}
.source-title {
    font-family: 'DM Serif Display', serif;
    font-size: 16px;
    color: #1A1814;
    line-height: 1.35;
    flex: 1;
}
.rrf-chip {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #8C8680;
    background: #EDEAE4;
    border: 1px solid #DDD9D2;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 2px;
}
.source-meta {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #8C8680;
    margin-bottom: 0.65rem;
}
.source-meta a { color: #6A8CC7; text-decoration: none; }
.source-meta a:hover { text-decoration: underline; }
.source-abstract {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 300;
    color: #8C8680;
    line-height: 1.65;
    border-top: 1px solid #E8E4DC;
    padding-top: 0.75rem;
}

/* ── Contradiction section ── */
.contra-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.contra-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8C8680;
}
.contra-body {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 300;
    line-height: 1.8;
    color: #1A1814;
}

/* ── Divider ── */
hr { border-color: #E8E4DC !important; margin: 1.75rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #C8B89A !important; }

/* ── Alert ── */
.stAlert {
    background: #F7F5F0 !important;
    border: 1px solid #E8E4DC !important;
    border-radius: 8px !important;
    color: #8C8680 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── Section label used in sources / contradiction ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8C8680;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────
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
2. AGREEMENTS: 2-3 specific points of agreement.
3. DISAGREEMENTS: Specific conflicting findings, or "None found."
Reference papers by [N]."""

# ── Session state ─────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "fill_query" not in st.session_state: st.session_state.fill_query = ""

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

def rrf_merge(dense, sparse, k=RRF_K, top_k=TOP_K_FINAL):
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
    return rrf_merge(
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

def get_answer(query, papers, client):
    ctx = build_context(papers)
    return ask_gemini(
        f"{RAG_SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{ctx}\n\n---\nQUESTION: {query}\n\nANSWER:", client
    )

def get_contradiction(query, papers, client):
    ctx = build_context(papers)
    return ask_gemini(CONTRADICTION_PROMPT.format(query=query, context=ctx), client)

def get_level(text):
    for l in ["HIGH","MODERATE","LOW","NONE"]:
        if l in text.upper(): return l
    return "UNKNOWN"

def level_pill(level):
    return f'<span class="lvl-pill lvl-{level}">{level}</span>'

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">ArXiv Lens</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand-sub">ML Research Assistant</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Gemini API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="Paste your key here", label_visibility="collapsed")
    if api_key:
        st.markdown('<span class="key-status-ok">✓ Key set — ready to search</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:12px;color:#8C8680;">Get a free key at <a href="https://aistudio.google.com" style="color:#6A8CC7;">aistudio.google.com</a></span>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-label">Knowledge Base</p>', unsafe_allow_html=True)
    for k, v in [
        ("Papers", "5,000"),
        ("Source", "ArXiv ML abstracts"),
        ("Categories", "cs.LG · cs.AI · cs.CL · cs.CV · stat.ML"),
        ("Embedding", "MiniLM-L6-v2 · 384d"),
        ("Chunk", "1 abstract = 1 chunk"),
    ]:
        st.markdown(f'<div class="sidebar-kv"><span class="sidebar-kv-k">{k}</span><span class="sidebar-kv-v">{v}</span></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-label">Retrieval Pipeline</p>', unsafe_allow_html=True)
    for step in ["Dense cosine similarity (top 10)", "BM25 keyword search (top 10)", "Reciprocal Rank Fusion (k=60)", f"Top {TOP_K_FINAL} papers → Gemini 3.5 Flash"]:
        st.markdown(f'<div class="pipeline-step"><div class="step-dot"></div><span class="step-text">{step}</span></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section-label">Recent Queries</p>', unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-5:]):
            short = q[:44] + "…" if len(q) > 44 else q
            st.markdown(f'<div class="history-pill" title="{q}">{short}</div>', unsafe_allow_html=True)

# ── Load indexes ──────────────────────────────────────────────
embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-eyebrow">5,000 ArXiv ML Papers · Hybrid Retrieval · Grounded by Gemini</div>
<h1 class="hero-heading">Ask the research.<br><em>Get cited answers.</em></h1>
<p class="hero-sub">
    Type a research question. ArXiv Lens retrieves the most relevant papers using both semantic
    and keyword search, then generates a grounded answer with inline citations —
    and tells you when papers disagree.
</p>
""", unsafe_allow_html=True)

# ── Search input ──────────────────────────────────────────────
# Pre-fill from example button click
default_query = st.session_state.fill_query
st.session_state.fill_query = ""

st.markdown('<div class="search-outer">', unsafe_allow_html=True)
query = st.text_input(
    "",
    value=default_query,
    placeholder="e.g. How does attention mechanism work in transformers?",
    label_visibility="collapsed",
    key="main_query"
)
st.markdown('</div>', unsafe_allow_html=True)

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    search_clicked = st.button("Search papers →", type="primary", disabled=not api_key or not query, use_container_width=True)
with col_hint:
    if not api_key:
        st.markdown('<p style="font-size:13px;color:#B8B0A4;padding-top:0.6rem;margin:0;">Add your Gemini API key in the sidebar to start.</p>', unsafe_allow_html=True)
    elif query:
        st.markdown('<p style="font-size:13px;color:#B8B0A4;padding-top:0.6rem;margin:0;">Press Enter or click Search.</p>', unsafe_allow_html=True)

# ── Example queries ───────────────────────────────────────────
EXAMPLES = [
    "What is transfer learning?",
    "How does reinforcement learning work?",
    "What are attention mechanisms?",
    "What is federated learning?",
    "How does BERT represent language?",
]

if not query:
    st.markdown('<p class="section-label" style="margin-top:1.25rem;">Try an example</p>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        with col:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.fill_query = ex
                st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Trigger logic ─────────────────────────────────────────────
# Fire on button click OR on Enter (query changed from last run)
trigger = search_clicked or (
    bool(query) and
    bool(api_key) and
    query != st.session_state.last_query
)

# ── Run pipeline ──────────────────────────────────────────────
if trigger and api_key and query:
    st.session_state.last_query = query
    if query not in st.session_state.history:
        st.session_state.history.append(query)

    client = genai.Client(api_key=api_key)

    with st.spinner("Scanning 5,000 papers via hybrid retrieval..."):
        papers = hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks)

    with st.spinner("Generating grounded answer with Gemini 3.5 Flash..."):
        answer = get_answer(query, papers, client)

    with st.spinner("Running cross-paper contradiction analysis..."):
        report = get_contradiction(query, papers, client)

    level = get_level(report)

    # Stats bar
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-cell">
            <div class="stat-label">Papers retrieved</div>
            <div class="stat-value gold">{len(papers)}</div>
            <div class="stat-sub">of 5,000 indexed</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Retrieval</div>
            <div class="stat-value" style="font-size:15px;padding-top:4px;">Dense + BM25</div>
            <div class="stat-sub">RRF fusion (k=60)</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Top RRF score</div>
            <div class="stat-value gold" style="font-size:18px;padding-top:4px;">{papers[0]['rrf_score']}</div>
            <div class="stat-sub">strongest match</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Contradiction</div>
            <div style="margin-top:6px;">{level_pill(level)}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Model</div>
            <div class="stat-value" style="font-size:14px;padding-top:4px;">Gemini 3.5 Flash</div>
            <div class="stat-sub">grounded generation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_a, tab_s, tab_c = st.tabs([
        "📝  Answer",
        f"📄  Sources  ({len(papers)})",
        f"⚖️  Contradictions  ·  {level}"
    ])

    # ── Answer tab
    with tab_a:
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="grounding-note">
            🔒 Every claim is grounded in the retrieved papers. [1], [2], etc. refer to sources in the Sources tab.
            When papers conflict, disagreements are stated explicitly — never silently blended.
        </div>
        """, unsafe_allow_html=True)

    # ── Sources tab
    with tab_s:
        st.markdown('<p class="section-label">Ranked by Reciprocal Rank Fusion across dense and keyword retrieval. Higher score = both methods agreed.</p>', unsafe_allow_html=True)
        for i, p in enumerate(papers, 1):
            aid   = p["metadata"]["arxiv_id"]
            title = p["metadata"]["title"]
            cats  = p["metadata"]["categories"]
            rrf_s = p["rrf_score"]
            url   = f"https://arxiv.org/abs/{aid}"
            abst  = p["document"][:340]
            st.markdown(f"""
            <div class="source-card">
                <div class="source-top">
                    <span class="source-num">[ {i} ]</span>
                    <span class="source-title">{title}</span>
                    <span class="rrf-chip">RRF {rrf_s}</span>
                </div>
                <div class="source-meta">
                    <a href="{url}" target="_blank">arxiv.org/abs/{aid}</a> &nbsp;·&nbsp; {cats}
                </div>
                <div class="source-abstract">{abst}…</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Contradiction tab
    with tab_c:
        st.markdown(f'<p class="section-label">Gemini analyses the {len(papers)} retrieved papers for agreements and conflicts. Most RAG systems silently blend contradictions — ArXiv Lens surfaces them.</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="contra-header">
            <span class="contra-label">Overall contradiction level</span>
            {level_pill(level)}
        </div>
        <div class="contra-body">{report}</div>
        """, unsafe_allow_html=True)