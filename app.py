import re
import pickle
from collections import defaultdict

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from google import genai

st.set_page_config(
    page_title="ArXiv Lens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  DESIGN SYSTEM
#  Palette: Ink #0F0E0C · Slate #2C2B28 · Stone #6B6860
#           Sand #C8B89A · Cream #F5F2EC · White #FFFFFF
#  Type: "Space Grotesk" utility/UI · "DM Serif Display" hero only
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

/* ══ RESET ══════════════════════════════════════════════════ */
*, html, body, [class*="css"] {
    box-sizing: border-box;
}
html, body, .stApp, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #F5F2EC !important;
    color: #0F0E0C !important;
}
.block-container {
    padding: 0 2.5rem 4rem 2.5rem !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}
#MainMenu, footer { visibility: hidden !important; }
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    height: 0 !important;
}

/* ══ SIDEBAR ════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0F0E0C !important;
    border-right: none !important;
    min-width: 240px !important;
    max-width: 240px !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 28px 20px 24px 20px !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #F5F2EC !important;
}
/* Sidebar logo */
.sb-logo {
    font-family: 'DM Serif Display', serif !important;
    font-size: 18px !important;
    color: #FFFFFF !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 2px !important;
}
.sb-tagline {
    font-size: 10px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #6B6860 !important;
    margin-bottom: 24px !important;
}
/* Section labels */
.sb-label {
    font-size: 9px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #6B6860 !important;
    margin: 20px 0 8px 0 !important;
    font-weight: 500 !important;
}
/* Sidebar divider */
.sb-divider {
    border: none !important;
    border-top: 1px solid #2C2B28 !important;
    margin: 16px 0 !important;
}
/* API key input in sidebar */
[data-testid="stSidebar"] .stTextInput input {
    background: #1A1917 !important;
    border: 1px solid #2C2B28 !important;
    border-radius: 6px !important;
    color: #F5F2EC !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 10px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #C8B89A !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #4A4845 !important;
}
[data-testid="stSidebar"] .stTextInput label {
    font-size: 9px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #6B6860 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] {
    border: none !important;
    box-shadow: none !important;
}
/* Stat rows */
.sb-stat {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 7px 0;
    border-bottom: 1px solid #1A1917;
}
.sb-stat-label {
    font-size: 11px;
    color: #6B6860;
    font-weight: 400;
}
.sb-stat-val {
    font-size: 11px;
    color: #C8B89A;
    font-weight: 500;
    text-align: right;
}
/* Pipeline steps */
.sb-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #1A1917;
    font-size: 11px;
    color: #9B9890;
}
.sb-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #C8B89A;
    flex-shrink: 0;
}
/* Recent queries */
.sb-query-chip {
    background: #1A1917;
    border: 1px solid #2C2B28;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 11px;
    color: #9B9890;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: default;
}

/* ══ MAIN SEARCH INPUT ══════════════════════════════════════ */
div[data-testid="stTextInputRootElement"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #E0DDD6 !important;
    border-radius: 10px !important;
    color: #0F0E0C !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 400 !important;
    padding: 14px 18px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stTextInputRootElement"] input:focus {
    border-color: #C8B89A !important;
    box-shadow: 0 0 0 3px rgba(200,184,154,0.18) !important;
    outline: none !important;
}
div[data-testid="stTextInputRootElement"] input::placeholder {
    color: #B0ADA6 !important;
}
div[data-testid="stTextInputRootElement"] {
    border: none !important;
    box-shadow: none !important;
}

/* ══ BUTTONS ════════════════════════════════════════════════ */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"] {
    background: #0F0E0C !important;
    color: #F5F2EC !important;
    border: none !important;
    padding: 11px 28px !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: #2C2B28 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(15,14,12,0.18) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: #E0DDD6 !important;
    color: #B0ADA6 !important;
}
/* Example chip buttons */
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #2C2B28 !important;
    border: 1px solid #E0DDD6 !important;
    padding: 7px 12px !important;
    font-size: 12px !important;
    border-radius: 20px !important;
    font-weight: 400 !important;
    white-space: normal !important;
    text-align: center !important;
    line-height: 1.35 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #0F0E0C !important;
    color: #F5F2EC !important;
    border-color: #0F0E0C !important;
    transform: translateY(-1px) !important;
}

/* ══ TABS ═══════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1.5px solid #E0DDD6 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #9B9890 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 22px !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: 0.01em !important;
}
.stTabs [aria-selected="true"] {
    color: #0F0E0C !important;
    border-bottom: 2px solid #0F0E0C !important;
    font-weight: 500 !important;
}

/* ══ SPINNER ════════════════════════════════════════════════ */
.stSpinner > div { border-top-color: #C8B89A !important; }

/* ══ ALERTS ═════════════════════════════════════════════════ */
.stAlert {
    background: #FFFFFF !important;
    border: 1px solid #E0DDD6 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
}

/* ══ MARKDOWN ═══════════════════════════════════════════════ */
.stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0F0E0C !important;
}

/* ══ FORCE SIDEBAR VISIBLE ══════════════════════════════════ */
[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 240px !important;
    margin-left: 0 !important;
    transform: translateX(0) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
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
2. If sources CONFLICT write: "Sources disagree: [A] says X, [B] says Y." Never blend conflicting claims.
3. If not in context: "Not covered in the retrieved papers."
4. Max 300 words. End with ### Sources listing titles cited."""

CONTRADICTION_PROMPT = """You are a research analyst comparing ML papers.
Given these papers for the query "{query}":
{context}

Respond with:
1. CONTRADICTION LEVEL: NONE / LOW / MODERATE / HIGH
2. AGREEMENTS: 2-3 specific points of agreement.
3. DISAGREEMENTS: Specific conflicting findings, or "None found."
Reference papers by [N]."""

# ── Session state ─────────────────────────────────────────────
for k, v in [("history", []), ("last_query", ""), ("pending_query", ""), ("trigger_search", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Resources ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_embeddings():
    return np.load(EMBEDDINGS_PATH)

@st.cache_resource(show_spinner="Loading chunks…")
def load_chunks():
    with open(CHUNKS_PATH, "rb") as f: return pickle.load(f)

@st.cache_resource(show_spinner="Loading BM25 index…")
def load_bm25():
    with open(BM25_PATH, "rb") as f: return pickle.load(f)

# ── Retrieval ─────────────────────────────────────────────────
def tokenise(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def dense_retrieve(query, embed_model, embeddings, chunks, top_k=TOP_K_DENSE):
    scores  = cosine_similarity(embed_model.encode([query]), embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"chunk_id": chunks[i]["chunk_id"], "score": float(scores[i]),
             "document": chunks[i]["chunk_text"],
             "metadata": {k: chunks[i][k] for k in ("arxiv_id","title","categories")}}
            for i in top_idx]

def sparse_retrieve(query, bm25_index, chunks, top_k=TOP_K_SPARSE):
    scores  = bm25_index.get_scores(tokenise(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"chunk_id": chunks[i]["chunk_id"], "score": float(scores[i]),
             "document": chunks[i]["chunk_text"],
             "metadata": {k: chunks[i][k] for k in ("arxiv_id","title","categories")}}
            for i in top_idx]

def rrf_merge(dense, sparse, k=RRF_K, top_k=TOP_K_FINAL):
    scores, cmap = defaultdict(float), {}
    for rank, h in enumerate(dense, 1):
        scores[h["chunk_id"]] += 1/(k+rank); cmap[h["chunk_id"]] = h
    for rank, h in enumerate(sparse, 1):
        scores[h["chunk_id"]] += 1/(k+rank)
        if h["chunk_id"] not in cmap: cmap[h["chunk_id"]] = h
    result = []
    for cid, sc in sorted(scores.items(), key=lambda x:-x[1])[:top_k]:
        h = dict(cmap[cid]); h["rrf_score"] = round(sc, 6); result.append(h)
    return result

def hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks):
    return rrf_merge(
        dense_retrieve(query, embed_model, embeddings, chunks),
        sparse_retrieve(query, bm25_index, chunks)
    )

# ── Gemini helpers ────────────────────────────────────────────
def build_context(papers):
    return "\n\n".join(
        f"[{i}] Title: {p['metadata']['title']}\n    ArXiv: {p['metadata']['arxiv_id']}\n    Text: {p['document']}"
        for i, p in enumerate(papers, 1)
    )

def ask_gemini(prompt, client):
    try:
        return client.models.generate_content(model=MODEL, contents=prompt).text
    except Exception as e:
        return f"**Error:** {e}"

def get_answer(query, papers, client):
    return ask_gemini(f"{RAG_SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{build_context(papers)}\n\n---\nQUESTION: {query}\n\nANSWER:", client)

def get_contradiction(query, papers, client):
    return ask_gemini(CONTRADICTION_PROMPT.format(query=query, context=build_context(papers)), client)

def get_level(text):
    for l in ["HIGH","MODERATE","LOW","NONE"]:
        if l in text.upper(): return l
    return "UNKNOWN"

LEVEL_META = {
    "NONE":     {"color":"#1A6641", "bg":"#D6F0E0", "border":"#A8D8BC"},
    "LOW":      {"color":"#1A4FA0", "bg":"#DBE9FF", "border":"#ACCBF5"},
    "MODERATE": {"color":"#7D4E00", "bg":"#FFF3CD", "border":"#F0D080"},
    "HIGH":     {"color":"#8B1A1A", "bg":"#FFE4E4", "border":"#F5AAAA"},
    "UNKNOWN":  {"color":"#6B6860", "bg":"#EDEAE4", "border":"#D8D5CE"},
}

def level_pill(level):
    m = LEVEL_META.get(level, LEVEL_META["UNKNOWN"])
    return (f'<span style="display:inline-flex;align-items:center;font-family:Space Grotesk,sans-serif;'
            f'font-size:11px;font-weight:600;letter-spacing:0.08em;padding:3px 10px;border-radius:20px;'
            f'color:{m["color"]};background:{m["bg"]};border:1px solid {m["border"]};">{level}</span>')

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-logo">ArXiv Lens</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">ML Research Assistant</div>', unsafe_allow_html=True)

    api_key = st.text_input("API Key", type="password", placeholder="Gemini API key…", label_visibility="visible")

    if api_key:
        st.markdown('<div style="font-size:11px;color:#4CAF82;background:#0A2218;border:1px solid #1A4030;'
                    'border-radius:5px;padding:5px 9px;margin-top:2px;">✓ Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:11px;color:#6B6860;margin-top:4px;">→ <a href="https://aistudio.google.com" '
                    'target="_blank" style="color:#C8B89A;">Get a free key</a></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)

    kb_rows = [
        ("Papers", "5,000"),
        ("Source", "ArXiv"),
        ("Topics", "cs.LG · cs.AI · cs.CL"),
        ("Embed model", "MiniLM-L6-v2"),
        ("Embed dim", "384"),
    ]
    for label, val in kb_rows:
        st.markdown(f'<div class="sb-stat"><span class="sb-stat-label">{label}</span>'
                    f'<span class="sb-stat-val">{val}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Pipeline</div>', unsafe_allow_html=True)
    for step in ["Dense cosine (top 10)", "BM25 keyword (top 10)", "RRF fusion k=60", f"Top {TOP_K_FINAL} → Gemini"]:
        st.markdown(f'<div class="sb-step"><div class="sb-dot"></div>{step}</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-label">Recent</div>', unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-4:]):
            short = (q[:32] + "…") if len(q) > 32 else q
            st.markdown(f'<div class="sb-query-chip" title="{q}">{short}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD INDEXES
# ══════════════════════════════════════════════════════════════
embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 52px 0 32px 0;">
  <p style="font-family:'Space Grotesk',sans-serif;font-size:10px;font-weight:500;
     letter-spacing:0.2em;text-transform:uppercase;color:#C8B89A;margin:0 0 16px 0;">
    5,000 ArXiv ML Papers · Hybrid Retrieval · Gemini
  </p>
  <h1 style="font-family:'DM Serif Display',serif;font-size:clamp(40px,4.5vw,58px);
     font-weight:400;color:#0F0E0C;line-height:1.1;margin:0 0 2px 0;letter-spacing:-0.02em;">
    Ask the research.
  </h1>
  <h1 style="font-family:'DM Serif Display',serif;font-style:italic;
     font-size:clamp(40px,4.5vw,58px);font-weight:400;color:#C8B89A;
     line-height:1.1;margin:0 0 20px 0;letter-spacing:-0.02em;">
    Get cited answers.
  </h1>
  <p style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:300;
     color:#6B6860;line-height:1.7;max-width:480px;margin:0;">
    Semantic search + keyword matching over 5,000 ML abstracts. 
    Every answer is grounded with inline citations. Contradictions are surfaced, never blended.
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SEARCH BAR — auto-populates from examples and triggers search
# ══════════════════════════════════════════════════════════════

# If an example was clicked, use its text as the prefill value
input_default = st.session_state.pending_query

query = st.text_input(
    "search",
    value=input_default,
    placeholder="e.g. How does the attention mechanism work in transformers?",
    label_visibility="collapsed",
    key="main_query_input"
)

# ── Search button + hint ───────────────────────────────────────
c1, c2 = st.columns([1, 5])
with c1:
    do_search = st.button("Search →", type="primary", disabled=(not api_key or not query), use_container_width=True)
with c2:
    if not api_key:
        st.markdown('<p style="font-size:13px;color:#B0ADA6;margin:8px 0 0 2px;">Add your Gemini API key in the sidebar.</p>', unsafe_allow_html=True)
    elif not query:
        st.markdown('<p style="font-size:13px;color:#B0ADA6;margin:8px 0 0 2px;">Type a question or pick an example below.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:13px;color:#B0ADA6;margin:8px 0 0 2px;">Press Enter or click Search.</p>', unsafe_allow_html=True)

# ── Example chips — one click populates AND searches ──────────
EXAMPLES = [
    "What is transfer learning?",
    "How does reinforcement learning work?",
    "What are attention mechanisms?",
    "What is federated learning?",
    "How does BERT represent language?",
]

st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:10px;font-weight:500;'
            'text-transform:uppercase;letter-spacing:0.16em;color:#9B9890;margin:20px 0 8px;">Try an example</p>',
            unsafe_allow_html=True)

ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    with col:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            # Store the example as the pending query AND mark it to trigger search
            st.session_state.pending_query = ex
            st.session_state.trigger_search = True
            st.rerun()

st.markdown('<hr style="border:none;border-top:1px solid #E0DDD6;margin:24px 0 28px;">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DETERMINE IF WE SHOULD RUN
# Examples set trigger_search=True, then on rerun the query
# field is pre-populated via pending_query, and we fire immediately.
# ══════════════════════════════════════════════════════════════
should_run = False

if st.session_state.trigger_search and st.session_state.pending_query:
    # Example was clicked — use the stored pending query, fire search
    effective_query = st.session_state.pending_query
    st.session_state.trigger_search = False
    st.session_state.pending_query  = ""
    should_run = True
elif do_search and query:
    effective_query = query
    should_run = True
elif query and api_key and query != st.session_state.last_query:
    # Enter was pressed
    effective_query = query
    should_run = True
else:
    effective_query = query

# ══════════════════════════════════════════════════════════════
# SEARCH PIPELINE
# ══════════════════════════════════════════════════════════════
if should_run and api_key and effective_query:
    st.session_state.last_query = effective_query
    if effective_query not in st.session_state.history:
        st.session_state.history.append(effective_query)

    client = genai.Client(api_key=api_key)

    prog_container = st.empty()
    def prog(msg):
        prog_container.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;'
            f'background:#FFFFFF;border:1px solid #E0DDD6;border-radius:8px;'
            f'font-family:Space Grotesk,sans-serif;font-size:13px;color:#6B6860;">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:#C8B89A;animation:pulse 1s infinite;"></span>{msg}</div>',
            unsafe_allow_html=True
        )

    prog("Scanning 5,000 papers via hybrid retrieval…")
    papers = hybrid_retrieve(effective_query, embed_model, embeddings, bm25_index, chunks)

    prog("Generating grounded answer with Gemini…")
    answer = get_answer(effective_query, papers, client)

    prog("Analysing papers for contradictions…")
    report = get_contradiction(effective_query, papers, client)

    prog_container.empty()

    level = get_level(report)
    lm    = LEVEL_META.get(level, LEVEL_META["UNKNOWN"])

    # ── Stats strip ───────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:0;background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;
         overflow:hidden;margin-bottom:28px;">
      <div style="flex:1;padding:16px 20px;border-right:1px solid #E0DDD6;">
        <div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;color:#9B9890;margin-bottom:6px;font-weight:500;">Papers retrieved</div>
        <div style="font-family:'DM Serif Display',serif;font-size:30px;color:#C8B89A;line-height:1;">{len(papers)}</div>
        <div style="font-size:11px;color:#B0ADA6;margin-top:2px;">of 5,000</div>
      </div>
      <div style="flex:1;padding:16px 20px;border-right:1px solid #E0DDD6;">
        <div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;color:#9B9890;margin-bottom:6px;font-weight:500;">Retrieval</div>
        <div style="font-size:14px;font-weight:500;color:#0F0E0C;margin-top:4px;">Dense + BM25</div>
        <div style="font-size:11px;color:#B0ADA6;margin-top:2px;">RRF fusion k=60</div>
      </div>
      <div style="flex:1;padding:16px 20px;border-right:1px solid #E0DDD6;">
        <div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;color:#9B9890;margin-bottom:6px;font-weight:500;">Top RRF score</div>
        <div style="font-family:'DM Serif Display',serif;font-size:30px;color:#C8B89A;line-height:1;">{papers[0]["rrf_score"]}</div>
        <div style="font-size:11px;color:#B0ADA6;margin-top:2px;">strongest match</div>
      </div>
      <div style="flex:1;padding:16px 20px;border-right:1px solid #E0DDD6;">
        <div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;color:#9B9890;margin-bottom:8px;font-weight:500;">Contradiction</div>
        {level_pill(level)}
      </div>
      <div style="flex:1;padding:16px 20px;">
        <div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;color:#9B9890;margin-bottom:6px;font-weight:500;">Model</div>
        <div style="font-size:13px;font-weight:500;color:#0F0E0C;margin-top:4px;">Gemini 2.0 Flash</div>
        <div style="font-size:11px;color:#B0ADA6;margin-top:2px;">grounded generation</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab_a, tab_s, tab_c = st.tabs([
        "Answer",
        f"Sources ({len(papers)})",
        f"Contradictions · {level}",
    ])

    with tab_a:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;
             padding:28px 32px;font-family:'Space Grotesk',sans-serif;font-size:15px;
             font-weight:300;line-height:1.8;color:#0F0E0C;margin-bottom:12px;">
          {answer}
        </div>
        <div style="background:#F5F2EC;border:1px solid #E0DDD6;border-radius:7px;
             padding:10px 14px;font-family:'Space Grotesk',sans-serif;font-size:12px;
             color:#9B9890;line-height:1.5;">
          🔒 Every claim is grounded in retrieved papers. [1], [2]… refer to sources in the Sources tab.
          Contradictions are stated explicitly — never silently blended.
        </div>
        """, unsafe_allow_html=True)

    with tab_s:
        st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:13px;color:#9B9890;'
                    'margin-bottom:16px;">Ranked by Reciprocal Rank Fusion across dense and keyword retrieval.</p>',
                    unsafe_allow_html=True)
        for i, p in enumerate(papers, 1):
            aid, title, cats = p["metadata"]["arxiv_id"], p["metadata"]["title"], p["metadata"]["categories"]
            url  = f"https://arxiv.org/abs/{aid}"
            abst = p["document"][:320]
            rrf_s = p["rrf_score"]
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;
                 padding:18px 22px;margin-bottom:10px;">
              <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:8px;">
                <span style="font-family:'Space Grotesk',sans-serif;font-size:10px;font-weight:600;
                   letter-spacing:0.08em;color:#C8B89A;background:#F5F2EC;border:1px solid #E0DDD6;
                   padding:3px 7px;border-radius:4px;flex-shrink:0;margin-top:2px;">[{i}]</span>
                <span style="font-family:'DM Serif Display',serif;font-size:16px;color:#0F0E0C;
                   line-height:1.35;flex:1;">{title}</span>
                <span style="font-family:'Space Grotesk',sans-serif;font-size:10px;color:#9B9890;
                   background:#F5F2EC;border:1px solid #E0DDD6;padding:3px 7px;border-radius:4px;
                   flex-shrink:0;margin-top:2px;white-space:nowrap;">RRF {rrf_s}</span>
              </div>
              <div style="font-size:11px;color:#9B9890;margin-bottom:10px;">
                <a href="{url}" target="_blank" style="color:#6A8CC7;text-decoration:none;">
                  arxiv.org/abs/{aid}</a> &nbsp;·&nbsp; {cats}
              </div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:300;
                   color:#6B6860;line-height:1.7;border-top:1px solid #E0DDD6;padding-top:12px;">
                {abst}…
              </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_c:
        st.markdown(f'<p style="font-family:Space Grotesk,sans-serif;font-size:13px;color:#9B9890;'
                    f'margin-bottom:16px;">Gemini cross-analyses the {len(papers)} retrieved papers for '
                    f'agreements and conflicts. Contradictions are surfaced, not hidden.</p>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:#FFFFFF;border:1px solid #E0DDD6;
             border-radius:8px;padding:12px 16px;margin-bottom:12px;">
          <span style="font-family:'Space Grotesk',sans-serif;font-size:11px;color:#9B9890;
             text-transform:uppercase;letter-spacing:0.1em;">Contradiction level</span>
          {level_pill(level)}
        </div>
        <div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;
             padding:22px 26px;font-family:'Space Grotesk',sans-serif;font-size:14px;
             font-weight:300;line-height:1.85;color:#0F0E0C;">
          {report}
        </div>
        """, unsafe_allow_html=True)