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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ════════════════════════════════════════
   BASE
════════════════════════════════════════ */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F7F5F0 !important;
    color: #1A1814 !important;
}
.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 1080px !important;
}
#MainMenu, footer, header { visibility: hidden !important; }

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E8E4DC !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: #1A1814 !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background-color: #F7F5F0 !important;
    border: 1px solid #DDD9D2 !important;
    border-radius: 8px !important;
    color: #1A1814 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #C8B89A !important;
    box-shadow: 0 0 0 3px rgba(200,184,154,0.15) !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #B8B0A4 !important;
}
[data-testid="stSidebar"] .stTextInput label {
    font-size: 12px !important;
    color: #8C8680 !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #1A1814 !important;
}

/* ════════════════════════════════════════
   MAIN SEARCH INPUT
════════════════════════════════════════ */
.main-query .stTextInput input,
div[data-testid="stTextInputRootElement"] input {
    background-color: #FFFFFF !important;
    border: 1.5px solid #DDD9D2 !important;
    border-radius: 12px !important;
    color: #1A1814 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 400 !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stTextInputRootElement"] input:focus {
    border-color: #C8B89A !important;
    box-shadow: 0 0 0 4px rgba(200,184,154,0.15) !important;
    outline: none !important;
}
div[data-testid="stTextInputRootElement"] input::placeholder {
    color: #B8B0A4 !important;
}
div[data-testid="stTextInputRootElement"] {
    border: none !important;
    box-shadow: none !important;
}

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"] {
    background-color: #1A1814 !important;
    color: #F7F5F0 !important;
    border: none !important;
    padding: 10px 24px !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2D2922 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(26,24,20,0.2) !important;
}
.stButton > button[kind="primary"]:disabled {
    background-color: #EDEAE4 !important;
    color: #B8B0A4 !important;
    transform: none !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"] {
    background-color: #EDEAE4 !important;
    color: #1A1814 !important;
    border: 1px solid #DDD9D2 !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #FFFFFF !important;
    border-color: #C8B89A !important;
}

/* ════════════════════════════════════════
   TABS
════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E8E4DC !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    color: #8C8680 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #1A1814 !important;
    border-bottom: 2px solid #C8B89A !important;
    font-weight: 500 !important;
}

/* ════════════════════════════════════════
   SPINNER
════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: #C8B89A !important;
}

/* ════════════════════════════════════════
   ALERTS
════════════════════════════════════════ */
.stAlert {
    background: #FAF8F5 !important;
    border: 1px solid #E8E4DC !important;
    border-radius: 8px !important;
    color: #8C8680 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ════════════════════════════════════════
   MARKDOWN TEXT OVERRIDE
════════════════════════════════════════ */
.stMarkdown p, .stMarkdown span, .stMarkdown li {
    color: #1A1814 !important;
    font-family: 'DM Sans', sans-serif !important;
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
if "history"    not in st.session_state: st.session_state.history    = []
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "fill_query" not in st.session_state: st.session_state.fill_query = ""
if "run_now"    not in st.session_state: st.session_state.run_now    = False

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

@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

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
        return f"**Error:** {e}"

def get_answer(query, papers, client):
    return ask_gemini(
        f"{RAG_SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{build_context(papers)}\n\n---\nQUESTION: {query}\n\nANSWER:",
        client
    )

def get_contradiction(query, papers, client):
    return ask_gemini(CONTRADICTION_PROMPT.format(query=query, context=build_context(papers)), client)

def get_level(text):
    for l in ["HIGH","MODERATE","LOW","NONE"]:
        if l in text.upper(): return l
    return "UNKNOWN"

LEVEL_STYLES = {
    "NONE":     ("color:#2D6A4F; background:#D8F3DC; border:1px solid #B7E4C7;"),
    "LOW":      ("color:#1A4FA0; background:#DBE9FF; border:1px solid #BDD0F0;"),
    "MODERATE": ("color:#7D4E00; background:#FFF3CD; border:1px solid #FFD97D;"),
    "HIGH":     ("color:#7D1A1A; background:#FFE5E5; border:1px solid #FFBABA;"),
    "UNKNOWN":  ("color:#5A5550; background:#EDEAE4; border:1px solid #DDD9D2;"),
}

def level_pill(level):
    style = LEVEL_STYLES.get(level, LEVEL_STYLES["UNKNOWN"])
    return f'<span style="display:inline-block;font-family:DM Sans,sans-serif;font-size:12px;font-weight:600;letter-spacing:0.06em;padding:4px 12px;border-radius:20px;{style}">{level}</span>'

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("### ArXiv Lens")
    st.markdown('<p style="font-size:11px;color:#8C8680;text-transform:uppercase;letter-spacing:0.1em;margin-top:-10px;margin-bottom:20px;">ML Research Assistant</p>', unsafe_allow_html=True)

    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your key here")
    if api_key:
        st.markdown('<p style="font-size:12px;color:#2D6A4F;background:#D8F3DC;border:1px solid #B7E4C7;border-radius:6px;padding:6px 10px;margin-top:4px;">✓ Key set — ready to search</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:12px;color:#8C8680;margin-top:4px;">Get a free key at <a href="https://aistudio.google.com" style="color:#6A8CC7;">aistudio.google.com</a></p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:#8C8680;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Knowledge Base</p>', unsafe_allow_html=True)

    kb_data = [
        ("Papers indexed", "5,000"),
        ("Source", "ArXiv ML abstracts"),
        ("Categories", "cs.LG, cs.AI, cs.CL, cs.CV, stat.ML"),
        ("Embedding model", "all-MiniLM-L6-v2"),
        ("Embedding dim", "384"),
        ("Chunk strategy", "1 abstract = 1 chunk"),
    ]
    for label, val in kb_data:
        st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #E8E4DC;"><span style="font-size:13px;color:#8C8680;">{label}</span><span style="font-size:13px;color:#1A1814;font-weight:500;text-align:right;max-width:55%;">{val}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:#8C8680;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Retrieval Pipeline</p>', unsafe_allow_html=True)

    steps = [
        "Dense cosine similarity (top 10)",
        "BM25 keyword search (top 10)",
        "Reciprocal Rank Fusion (k=60)",
        f"Top {TOP_K_FINAL} papers → Gemini",
    ]
    for step in steps:
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #E8E4DC;"><div style="width:6px;height:6px;border-radius:50%;background:#C8B89A;flex-shrink:0;"></div><span style="font-size:13px;color:#1A1814;">{step}</span></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown('<p style="font-size:11px;color:#8C8680;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Recent Queries</p>', unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-5:]):
            short = q[:42] + "…" if len(q) > 42 else q
            st.markdown(f'<div style="font-size:12px;color:#8C8680;background:#EDEAE4;border:1px solid #DDD9D2;border-radius:6px;padding:5px 9px;margin-bottom:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{q}">{short}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════
# LOAD INDEXES
# ════════════════════════════════════════
embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ════════════════════════════════════════
# HERO
# ════════════════════════════════════════
st.markdown("""
<p style="font-family:'DM Sans',sans-serif;font-size:11px;font-weight:400;letter-spacing:0.16em;text-transform:uppercase;color:#C8B89A;margin-bottom:12px;">
    5,000 ArXiv ML Papers &nbsp;·&nbsp; Hybrid Retrieval &nbsp;·&nbsp; Grounded by Gemini
</p>
<h1 style="font-family:'DM Serif Display',serif;font-size:clamp(36px,5vw,56px);font-weight:400;color:#1A1814;line-height:1.12;margin:0 0 4px 0;letter-spacing:-0.01em;">
    Ask the research.
</h1>
<h1 style="font-family:'DM Serif Display',serif;font-style:italic;font-size:clamp(36px,5vw,56px);font-weight:400;color:#C8B89A;line-height:1.12;margin:0 0 20px 0;letter-spacing:-0.01em;">
    Get cited answers.
</h1>
<p style="font-family:'DM Sans',sans-serif;font-size:16px;font-weight:300;color:#8C8680;line-height:1.65;max-width:520px;margin:0 0 2rem 0;">
    Type a research question. ArXiv Lens retrieves the most relevant papers using both
    semantic and keyword search, then generates a grounded answer with inline citations —
    and tells you when papers disagree.
</p>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════
default_val = st.session_state.fill_query
st.session_state.fill_query = ""

query = st.text_input(
    "Your question",
    value=default_val,
    placeholder="e.g. How does the attention mechanism work in transformers?",
    label_visibility="collapsed",
    key="main_query"
)

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    search_clicked = st.button(
        "Search papers →",
        type="primary",
        disabled=not api_key or not query,
        use_container_width=True
    )
with col_hint:
    if not api_key:
        st.markdown('<p style="font-size:13px;color:#B8B0A4;padding-top:8px;margin:0;">Add your Gemini API key in the sidebar to start.</p>', unsafe_allow_html=True)
    elif not query:
        st.markdown('<p style="font-size:13px;color:#B8B0A4;padding-top:8px;margin:0;">Type a question above, then press Enter or click Search.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:13px;color:#B8B0A4;padding-top:8px;margin:0;">Press Enter or click Search.</p>', unsafe_allow_html=True)

# ── Example queries ───────────────────────────────────────────
EXAMPLES = [
    "What is transfer learning?",
    "How does reinforcement learning work?",
    "What are attention mechanisms?",
    "What is federated learning?",
    "How does BERT represent language?",
]

if not query:
    st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.12em;color:#8C8680;margin:20px 0 10px;">Try an example</p>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        with col:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.fill_query = ex
                st.session_state.run_now    = True
                st.rerun()

st.markdown('<hr style="border:none;border-top:1px solid #E8E4DC;margin:24px 0;">', unsafe_allow_html=True)

# ════════════════════════════════════════
# TRIGGER
# ════════════════════════════════════════
run_now = st.session_state.run_now
st.session_state.run_now = False

trigger = search_clicked or run_now or (
    bool(query) and bool(api_key) and query != st.session_state.last_query
)

# ════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════
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

    # ── Stats bar
    st.markdown(f"""
    <div style="display:flex;background:#FFFFFF;border:1px solid #E8E4DC;border-radius:10px;overflow:hidden;margin-bottom:24px;">
        <div style="flex:1;padding:14px 18px;border-right:1px solid #E8E4DC;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.1em;color:#8C8680;margin-bottom:4px;">Papers retrieved</div>
            <div style="font-family:DM Serif Display,serif;font-size:28px;color:#C8B89A;">{len(papers)}</div>
            <div style="font-family:DM Sans,sans-serif;font-size:12px;color:#8C8680;">of 5,000 indexed</div>
        </div>
        <div style="flex:1;padding:14px 18px;border-right:1px solid #E8E4DC;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.1em;color:#8C8680;margin-bottom:4px;">Retrieval method</div>
            <div style="font-family:DM Sans,sans-serif;font-size:15px;font-weight:500;color:#1A1814;margin-top:6px;">Dense + BM25</div>
            <div style="font-family:DM Sans,sans-serif;font-size:12px;color:#8C8680;">RRF fusion (k=60)</div>
        </div>
        <div style="flex:1;padding:14px 18px;border-right:1px solid #E8E4DC;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.1em;color:#8C8680;margin-bottom:4px;">Top RRF score</div>
            <div style="font-family:DM Serif Display,serif;font-size:28px;color:#C8B89A;">{papers[0]['rrf_score']}</div>
            <div style="font-family:DM Sans,sans-serif;font-size:12px;color:#8C8680;">strongest match</div>
        </div>
        <div style="flex:1;padding:14px 18px;border-right:1px solid #E8E4DC;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.1em;color:#8C8680;margin-bottom:8px;">Contradiction</div>
            {level_pill(level)}
        </div>
        <div style="flex:1;padding:14px 18px;">
            <div style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.1em;color:#8C8680;margin-bottom:4px;">Generation model</div>
            <div style="font-family:DM Sans,sans-serif;font-size:14px;font-weight:500;color:#1A1814;margin-top:6px;">Gemini 3.5 Flash</div>
            <div style="font-family:DM Sans,sans-serif;font-size:12px;color:#8C8680;">grounded generation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs
    tab_a, tab_s, tab_c = st.tabs([
        "📝  Answer",
        f"📄  Sources  ({len(papers)})",
        f"⚖️  Contradictions  ·  {level}"
    ])

    # ── Answer
    with tab_a:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E8E4DC;border-radius:12px;padding:28px 32px;font-family:DM Sans,sans-serif;font-size:15px;font-weight:300;line-height:1.8;color:#1A1814;margin-bottom:12px;">
            {answer}
        </div>
        <div style="background:#FAF8F5;border:1px solid #E8E4DC;border-radius:8px;padding:10px 14px;font-family:DM Sans,sans-serif;font-size:13px;color:#8C8680;line-height:1.5;">
            🔒 Every claim is grounded in the retrieved papers. [1], [2], etc. refer to sources in the Sources tab.
            When papers conflict, disagreements are stated explicitly — never silently blended.
        </div>
        """, unsafe_allow_html=True)

    # ── Sources
    with tab_s:
        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:13px;color:#8C8680;margin-bottom:16px;">Ranked by Reciprocal Rank Fusion across dense semantic and BM25 keyword retrieval. Higher score = stronger agreement across both methods.</p>', unsafe_allow_html=True)

        for i, p in enumerate(papers, 1):
            aid   = p["metadata"]["arxiv_id"]
            title = p["metadata"]["title"]
            cats  = p["metadata"]["categories"]
            rrf_s = p["rrf_score"]
            url   = f"https://arxiv.org/abs/{aid}"
            abst  = p["document"][:340]
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #E8E4DC;border-radius:10px;padding:18px 22px;margin-bottom:12px;transition:border-color 0.15s;">
                <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:8px;">
                    <span style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.08em;color:#C8B89A;background:#FAF8F5;border:1px solid #E8E4DC;padding:3px 8px;border-radius:4px;flex-shrink:0;margin-top:2px;">[ {i} ]</span>
                    <span style="font-family:DM Serif Display,serif;font-size:16px;color:#1A1814;line-height:1.35;flex:1;">{title}</span>
                    <span style="font-family:DM Sans,sans-serif;font-size:11px;color:#8C8680;background:#EDEAE4;border:1px solid #DDD9D2;padding:3px 8px;border-radius:4px;flex-shrink:0;margin-top:2px;">RRF {rrf_s}</span>
                </div>
                <div style="font-family:DM Sans,sans-serif;font-size:12px;color:#8C8680;margin-bottom:10px;">
                    <a href="{url}" target="_blank" style="color:#6A8CC7;text-decoration:none;">arxiv.org/abs/{aid}</a> &nbsp;·&nbsp; {cats}
                </div>
                <div style="font-family:DM Sans,sans-serif;font-size:13px;font-weight:300;color:#8C8680;line-height:1.65;border-top:1px solid #E8E4DC;padding-top:12px;">
                    {abst}…
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Contradictions
    with tab_c:
        st.markdown(f'<p style="font-family:DM Sans,sans-serif;font-size:13px;color:#8C8680;margin-bottom:16px;">Gemini analyses the {len(papers)} retrieved papers for agreements and conflicts. Most RAG systems silently blend contradictions — ArXiv Lens surfaces them explicitly.</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:#FFFFFF;border:1px solid #E8E4DC;border-radius:10px;padding:14px 18px;margin-bottom:10px;">
            <span style="font-family:DM Sans,sans-serif;font-size:11px;font-weight:400;text-transform:uppercase;letter-spacing:0.1em;color:#8C8680;">Overall contradiction level</span>
            {level_pill(level)}
        </div>
        <div style="background:#FFFFFF;border:1px solid #E8E4DC;border-radius:10px;padding:22px 26px;font-family:DM Sans,sans-serif;font-size:14px;font-weight:300;line-height:1.8;color:#1A1814;">
            {report}
        </div>
        """, unsafe_allow_html=True)