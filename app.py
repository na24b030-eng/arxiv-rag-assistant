import re
import pickle
from collections import defaultdict

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ArXiv Lens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
#  Ink #0F0E0C · Slate #2C2B28 · Stone #6B6860 · Sand #C8B89A
#  Cream #F5F2EC · White #FFFFFF · Border #E0DDD6 · Muted #9B9890
#  Inter (UI) · DM Serif Display (hero + paper titles only)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

*, html, body, [class*="css"] { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #F5F2EC !important;
    color: #0F0E0C !important;
}

/* Kill chrome — but keep the header element itself so sidebar toggle button stays */
#MainMenu, footer { visibility: hidden !important; }
header[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    height: 0 !important;
    overflow: hidden !important;
}

.block-container {
    padding: 40px 44px 80px 44px !important;
    max-width: 920px !important;
    margin: 0 auto !important;
}

/* ── SIDEBAR ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0F0E0C !important;
    border-right: none !important;
    min-width: 240px !important;
    max-width: 240px !important;
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 240px !important;
    margin-left: 0 !important;
    transform: translateX(0) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebarContent"] { padding: 28px 20px !important; }
[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
    color: #F5F2EC !important;
}
[data-testid="stSidebar"] .stTextInput label {
    font-size: 9px !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #6B6860 !important;
}
[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] {
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] input {
    background: #1A1917 !important;
    border: 1px solid #2C2B28 !important;
    border-radius: 7px !important;
    color: #F5F2EC !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 9px 11px !important;
}
[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] input:focus {
    border-color: #C8B89A !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] input::placeholder {
    color: #4A4845 !important;
}
/* Recent query re-run buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: #1A1917 !important;
    color: #9B9890 !important;
    border: 1px solid #2C2B28 !important;
    border-radius: 5px !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    padding: 5px 9px !important;
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100% !important;
    margin-bottom: 4px !important;
    transition: all 0.12s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2C2B28 !important;
    color: #C8B89A !important;
    border-color: #C8B89A !important;
    transform: none !important;
}

/* ── MAIN TEXT INPUT ──────────────────────────────────────── */
div[data-testid="stTextInputRootElement"] {
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stTextInputRootElement"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #E0DDD6 !important;
    border-radius: 10px !important;
    color: #0F0E0C !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    padding: 14px 18px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stTextInputRootElement"] input:focus {
    border-color: #C8B89A !important;
    box-shadow: 0 0 0 3px rgba(200,184,154,0.18) !important;
    outline: none !important;
}
div[data-testid="stTextInputRootElement"] input::placeholder { color: #B0ADA6 !important; }

/* ── BUTTONS ──────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
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
    box-shadow: 0 4px 14px rgba(15,14,12,0.2) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: #E0DDD6 !important;
    color: #B0ADA6 !important;
    cursor: not-allowed !important;
}
/* Example chips */
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #2C2B28 !important;
    border: 1px solid #E0DDD6 !important;
    padding: 7px 14px !important;
    font-size: 12px !important;
    border-radius: 20px !important;
    font-weight: 400 !important;
    white-space: normal !important;
    text-align: center !important;
    line-height: 1.4 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #0F0E0C !important;
    color: #F5F2EC !important;
    border-color: #0F0E0C !important;
    transform: translateY(-1px) !important;
}

/* Download buttons */
.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    background: #F5F2EC !important;
    color: #0F0E0C !important;
    border: 1px solid #E0DDD6 !important;
    border-radius: 7px !important;
    padding: 8px 14px !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: #0F0E0C !important;
    color: #F5F2EC !important;
    border-color: #0F0E0C !important;
}

/* ── TABS ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1.5px solid #E0DDD6 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #9B9890 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 22px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #0F0E0C !important;
    border-bottom: 2px solid #0F0E0C !important;
    font-weight: 500 !important;
}

/* ── SPINNER ──────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #C8B89A !important; }

/* ── MARKDOWN ─────────────────────────────────────────────── */
.stMarkdown p, .stMarkdown span, .stMarkdown li {
    font-family: 'Inter', sans-serif !important;
    color: #0F0E0C !important;
}

/* ── ALERT (error states) ─────────────────────────────────── */
.stAlert {
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
EMBEDDINGS_PATH = "embeddings.npy"
CHUNKS_PATH     = "chunks.pkl"
BM25_PATH       = "bm25_index.pkl"
TOP_K_DENSE  = 10
TOP_K_SPARSE = 10
TOP_K_FINAL  = 5
RRF_K        = 60
MODEL        = "gemini-3.5-flash"   # falls back gracefully in ask_gemini

# Single combined prompt — one Gemini call instead of two (halves latency + cost)
COMBINED_PROMPT = """You are a precise ML research assistant AND a research analyst.

Given these retrieved papers:
{context}

Answer the question: {query}

Respond with EXACTLY this structure — no deviations:

ANSWER:
<your answer here, max 280 words, citing every claim as [N] inline.
If a claim is not in context write "Not covered in the retrieved papers."
If sources conflict write "Sources disagree: [X] says A while [Y] says B."
End with a ### Sources section listing every title you cited.>

CONTRADICTION_LEVEL: <exactly one of: NONE / LOW / MODERATE / HIGH>

AGREEMENTS:
<2-3 specific points the papers agree on, referencing by [N]>

DISAGREEMENTS:
<specific conflicting findings referencing by [N], or "None found.">"""

EXAMPLES = [
    "What is transfer learning?",
    "How do transformers scale?",
    "What are attention mechanisms?",
    "What is federated learning?",
    "How does BERT represent language?",
]

# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
_defaults = {
    "history":        [],
    "last_query":     "",
    "pending_query":  "",
    "trigger_search": False,
    "results":        None,
    "search_error":   None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════
#  CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_embeddings():
    return np.load(EMBEDDINGS_PATH)

@st.cache_resource(show_spinner="Loading chunks…")
def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading BM25 index…")
def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

# ═══════════════════════════════════════════════════════════════
#  RETRIEVAL
# ═══════════════════════════════════════════════════════════════
def tokenise(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def dense_retrieve(query, embed_model, embeddings, chunks, top_k=TOP_K_DENSE):
    scores  = cosine_similarity(embed_model.encode([query]), embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "score":    float(scores[i]),
            "document": chunks[i]["chunk_text"],
            "metadata": {k: chunks[i][k] for k in ("arxiv_id", "title", "categories")},
        }
        for i in top_idx
    ]

def sparse_retrieve(query, bm25_index, chunks, top_k=TOP_K_SPARSE):
    scores  = bm25_index.get_scores(tokenise(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "score":    float(scores[i]),
            "document": chunks[i]["chunk_text"],
            "metadata": {k: chunks[i][k] for k in ("arxiv_id", "title", "categories")},
        }
        for i in top_idx
    ]

def rrf_merge(dense, sparse, k=RRF_K, top_k=TOP_K_FINAL):
    scores, cmap = defaultdict(float), {}
    for rank, h in enumerate(dense, 1):
        scores[h["chunk_id"]] += 1 / (k + rank)
        cmap[h["chunk_id"]] = h
    for rank, h in enumerate(sparse, 1):
        scores[h["chunk_id"]] += 1 / (k + rank)
        if h["chunk_id"] not in cmap:
            cmap[h["chunk_id"]] = h
    result = []
    for cid, sc in sorted(scores.items(), key=lambda x: -x[1])[:top_k]:
        h = dict(cmap[cid])
        h["rrf_score"] = round(sc, 6)
        result.append(h)
    return result

def hybrid_retrieve(query, embed_model, embeddings, bm25_index, chunks):
    return rrf_merge(
        dense_retrieve(query, embed_model, embeddings, chunks),
        sparse_retrieve(query, bm25_index, chunks),
    )

# ═══════════════════════════════════════════════════════════════
#  GEMINI — single call, structured parse
# ═══════════════════════════════════════════════════════════════
def build_context(papers):
    return "\n\n".join(
        f"[{i}] Title: {p['metadata']['title']}\n"
        f"    ArXiv: {p['metadata']['arxiv_id']}\n"
        f"    Text: {p['document']}"
        for i, p in enumerate(papers, 1)
    )

def run_gemini(query, papers, api_key):
    """
    Single Gemini call returning (answer, contradiction_level, agreements, disagreements).
    Raises ValueError with a clean message on API error.
    """
    client  = genai.Client(api_key=api_key)
    prompt  = COMBINED_PROMPT.format(context=build_context(papers), query=query)
    try:
        raw = client.models.generate_content(model=MODEL, contents=prompt).text
    except Exception as e:
        err = str(e)
        if "API_KEY" in err.upper() or "401" in err or "403" in err:
            raise ValueError("Invalid API key. Check your Gemini key in the sidebar.")
        if "404" in err or "not found" in err.lower():
            raise ValueError(f"Model '{MODEL}' not available. Try 'gemini-1.5-flash'.")
        raise ValueError(f"Gemini API error: {err}")

    # Parse structured sections
    def extract(tag, text):
        pattern = rf"{tag}:\s*(.*?)(?=\n[A-Z_]+:|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    answer       = extract("ANSWER", raw)
    level_raw    = extract("CONTRADICTION_LEVEL", raw)
    agreements   = extract("AGREEMENTS", raw)
    disagreements = extract("DISAGREEMENTS", raw)

    # Normalise level
    level = "UNKNOWN"
    for l in ["HIGH", "MODERATE", "LOW", "NONE"]:
        if l in level_raw.upper():
            level = l
            break

    # Fallback: if parsing fails, return raw text as answer
    if not answer:
        answer = raw

    return answer, level, agreements, disagreements

# ═══════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════
LEVEL_META = {
    "NONE":     {"color": "#166534", "bg": "#DCFCE7", "border": "#86EFAC"},
    "LOW":      {"color": "#1E40AF", "bg": "#DBEAFE", "border": "#93C5FD"},
    "MODERATE": {"color": "#92400E", "bg": "#FEF3C7", "border": "#FCD34D"},
    "HIGH":     {"color": "#991B1B", "bg": "#FEE2E2", "border": "#FCA5A5"},
    "UNKNOWN":  {"color": "#6B6860", "bg": "#F5F2EC", "border": "#E0DDD6"},
}

def level_badge(level):
    m = LEVEL_META.get(level, LEVEL_META["UNKNOWN"])
    return (
        f'<span style="display:inline-flex;align-items:center;font-family:Inter,sans-serif;'
        f'font-size:11px;font-weight:600;letter-spacing:0.07em;padding:3px 10px;border-radius:20px;'
        f'color:{m["color"]};background:{m["bg"]};border:1px solid {m["border"]};">{level}</span>'
    )

def stat_cell(label, value, subvalue="", border_right=True):
    border = "border-right:1px solid #E0DDD6;" if border_right else ""
    big = (
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:28px;'
        f'color:#C8B89A;line-height:1;">{value}</div>'
        if isinstance(value, (int, float)) or (isinstance(value, str) and value[0].isdigit())
        else f'<div style="font-size:14px;font-weight:500;color:#0F0E0C;margin-top:4px;'
             f'font-family:Inter,sans-serif;">{value}</div>'
    )
    return (
        f'<div style="flex:1;padding:16px 20px;{border}">'
        f'<div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;'
        f'color:#9B9890;margin-bottom:6px;font-weight:500;font-family:Inter,sans-serif;">{label}</div>'
        f'{big}'
        f'<div style="font-size:11px;color:#B0ADA6;margin-top:3px;font-family:Inter,sans-serif;">{subvalue}</div>'
        f'</div>'
    )

def sb_row(label, val):
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:7px 0;border-bottom:1px solid #1A1917;">'
        f'<span style="font-size:11px;color:#6B6860;font-family:Inter,sans-serif;">{label}</span>'
        f'<span style="font-size:11px;color:#C8B89A;font-weight:500;font-family:Inter,sans-serif;'
        f'text-align:right;">{val}</span></div>'
    )

def sb_step(text, n):
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
        f'border-bottom:1px solid #1A1917;">'
        f'<div style="width:16px;height:16px;border-radius:50%;background:#1A1917;border:1px solid #2C2B28;'
        f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
        f'<span style="font-size:9px;color:#C8B89A;font-weight:600;">{n}</span></div>'
        f'<span style="font-size:11px;color:#9B9890;font-family:Inter,sans-serif;">{text}</span></div>'
    )

def progress_bar(msg, step, total=3):
    pct = int((step / total) * 100)
    return (
        f'<div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:8px;padding:14px 18px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
        f'<span style="font-family:Inter,sans-serif;font-size:13px;color:#0F0E0C;">{msg}</span>'
        f'<span style="font-family:Inter,sans-serif;font-size:12px;color:#9B9890;">{step}/{total}</span>'
        f'</div>'
        f'<div style="background:#F5F2EC;border-radius:4px;height:3px;">'
        f'<div style="background:#C8B89A;height:3px;border-radius:4px;width:{pct}%;'
        f'transition:width 0.3s ease;"></div></div></div>'
    )

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:19px;color:#FFFFFF;'
        'margin:0 0 2px 0;letter-spacing:-0.01em;">ArXiv Lens</p>'
        '<p style="font-family:Inter,sans-serif;font-size:9px;letter-spacing:0.18em;'
        'text-transform:uppercase;color:#6B6860;margin:0 0 20px 0;">ML Research Assistant</p>',
        unsafe_allow_html=True,
    )

    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste key here…")

    if api_key:
        st.markdown(
            '<div style="font-size:11px;color:#4ADE80;background:#052E16;border:1px solid #166534;'
            'border-radius:5px;padding:5px 9px;margin-top:4px;font-family:Inter,sans-serif;">'
            '✓ Key set</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:11px;color:#6B6860;margin-top:4px;font-family:Inter,sans-serif;">'
            '→ <a href="https://aistudio.google.com" target="_blank" '
            'style="color:#C8B89A;text-decoration:none;">Get a free key</a></p>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border:none;border-top:1px solid #2C2B28;margin:18px 0;">', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:9px;letter-spacing:0.18em;text-transform:uppercase;color:#6B6860;'
        'margin:0 0 6px 0;font-family:Inter,sans-serif;font-weight:500;">Knowledge Base</p>',
        unsafe_allow_html=True,
    )
    for label, val in [
        ("Papers", "5,000"),
        ("Source", "ArXiv"),
        ("Topics", "cs.LG · cs.AI · cs.CL"),
        ("Embeddings", "MiniLM-L6"),
        ("Dimensions", "384"),
    ]:
        st.markdown(sb_row(label, val), unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:9px;letter-spacing:0.18em;text-transform:uppercase;color:#6B6860;'
        'margin:16px 0 6px 0;font-family:Inter,sans-serif;font-weight:500;">Pipeline</p>',
        unsafe_allow_html=True,
    )
    for n, step in enumerate(
        ["Dense cosine (top 10)", "BM25 keyword (top 10)", "RRF fusion k=60", f"Top {TOP_K_FINAL} → Gemini"],
        1,
    ):
        st.markdown(sb_step(step, n), unsafe_allow_html=True)

    # Recent queries — clickable, actually re-run the search
    if st.session_state.history:
        st.markdown('<hr style="border:none;border-top:1px solid #2C2B28;margin:18px 0;">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:9px;letter-spacing:0.18em;text-transform:uppercase;color:#6B6860;'
            'margin:0 0 8px 0;font-family:Inter,sans-serif;font-weight:500;">Recent — click to re-run</p>',
            unsafe_allow_html=True,
        )
        for q in reversed(st.session_state.history[-4:]):
            short = (q[:28] + "…") if len(q) > 28 else q
            if st.button(short, key=f"recent_{q}", help=q):
                st.session_state.pending_query  = q
                st.session_state.trigger_search = True
                st.rerun()

# ═══════════════════════════════════════════════════════════════
#  LOAD INDEXES
# ═══════════════════════════════════════════════════════════════
embed_model = load_embed_model()
embeddings  = load_embeddings()
chunks      = load_chunks()
bm25_index  = load_bm25()

# ═══════════════════════════════════════════════════════════════
#  HERO — only shown when no results yet
# ═══════════════════════════════════════════════════════════════
if not st.session_state.results:
    st.markdown("""
    <div style="padding:44px 0 32px 0;">
      <p style="font-family:Inter,sans-serif;font-size:10px;font-weight:500;
         letter-spacing:0.2em;text-transform:uppercase;color:#C8B89A;margin:0 0 14px 0;">
        5,000 ArXiv ML Papers · Hybrid Retrieval · Gemini Grounded
      </p>
      <h1 style="font-family:'DM Serif Display',serif;font-size:clamp(36px,4vw,52px);
         font-weight:400;color:#0F0E0C;line-height:1.1;margin:0;letter-spacing:-0.02em;">
        Ask the research.
      </h1>
      <h1 style="font-family:'DM Serif Display',serif;font-style:italic;
         font-size:clamp(36px,4vw,52px);font-weight:400;color:#C8B89A;
         line-height:1.1;margin:0 0 18px 0;letter-spacing:-0.02em;">
        Get cited answers.
      </h1>
      <p style="font-family:Inter,sans-serif;font-size:15px;font-weight:300;
         color:#6B6860;line-height:1.7;max-width:480px;margin:0;">
        Semantic + keyword search over 5,000 ML abstracts.
        Every answer is grounded with inline citations.
        Contradictions are surfaced — never silently blended.
      </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Compact header when results are showing — just the brand
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:20px;color:#0F0E0C;'
        'margin:12px 0 20px 0;letter-spacing:-0.01em;">🔭 ArXiv Lens</p>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
#  SEARCH INPUT
# ═══════════════════════════════════════════════════════════════
input_default = st.session_state.pending_query

query = st.text_input(
    "question",
    value=input_default,
    placeholder="e.g. How does the attention mechanism work in transformers?",
    label_visibility="collapsed",
    key="main_query_input",
)

c_btn, c_hint = st.columns([1, 5])
with c_btn:
    do_search = st.button(
        "Search →",
        type="primary",
        disabled=(not api_key or not query),
        use_container_width=True,
    )
with c_hint:
    if not api_key:
        st.markdown(
            '<p style="font-size:13px;color:#B0ADA6;margin:9px 0 0 4px;font-family:Inter,sans-serif;">'
            'Add your Gemini API key in the sidebar to start.</p>',
            unsafe_allow_html=True,
        )
    elif not query:
        st.markdown(
            '<p style="font-size:13px;color:#B0ADA6;margin:9px 0 0 4px;font-family:Inter,sans-serif;">'
            'Type a question or pick an example below.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:13px;color:#B0ADA6;margin:9px 0 0 4px;font-family:Inter,sans-serif;">'
            'Press Enter or click Search →</p>',
            unsafe_allow_html=True,
        )

# Example chips — only on home screen (no results yet)
if not st.session_state.results:
    st.markdown(
        '<p style="font-family:Inter,sans-serif;font-size:10px;font-weight:500;'
        'text-transform:uppercase;letter-spacing:0.16em;color:#9B9890;margin:22px 0 10px;">'
        'Try an example</p>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        with col:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.pending_query  = ex
                st.session_state.trigger_search = True
                st.rerun()

st.markdown('<hr style="border:none;border-top:1px solid #E0DDD6;margin:24px 0 28px;">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TRIGGER LOGIC
#  Path A — example chip or recent query → pending_query + trigger_search
#            → st.rerun() → next render fires pipeline
#  Path B — Search button clicked → immediate
#  Path C — Enter pressed (query changed, not equal to last_query)
#            → ONLY fires if api_key is set, prevents spurious rerenders
# ═══════════════════════════════════════════════════════════════
should_run      = False
effective_query = query

if st.session_state.trigger_search and st.session_state.pending_query:
    effective_query                 = st.session_state.pending_query
    st.session_state.trigger_search = False
    st.session_state.pending_query  = ""
    should_run = True
elif do_search and query:
    effective_query = query
    should_run = True
elif (
    query
    and api_key
    and query.strip() != st.session_state.last_query.strip()
    and not do_search          # Enter path only — not double-firing with button
    and len(query.strip()) > 4 # Ignore very short accidental triggers
):
    effective_query = query
    should_run = True

# ═══════════════════════════════════════════════════════════════
#  SEARCH PIPELINE
# ═══════════════════════════════════════════════════════════════
if should_run and api_key and effective_query:
    st.session_state.last_query   = effective_query.strip()
    st.session_state.search_error = None

    if effective_query not in st.session_state.history:
        st.session_state.history.append(effective_query)

    _prog = st.empty()

    _prog.markdown(progress_bar("Scanning 5,000 papers via hybrid retrieval…", 1), unsafe_allow_html=True)
    papers = hybrid_retrieve(effective_query, embed_model, embeddings, bm25_index, chunks)

    # Guard: empty results
    if not papers:
        _prog.empty()
        st.session_state.search_error = "No papers matched your query. Try rephrasing or a broader question."
    else:
        _prog.markdown(progress_bar("Generating answer and contradiction analysis…", 2), unsafe_allow_html=True)
        try:
            answer, level, agreements, disagreements = run_gemini(effective_query, papers, api_key)
            _prog.markdown(progress_bar("Done.", 3), unsafe_allow_html=True)
            import time; time.sleep(0.4)
            _prog.empty()

            st.session_state.results = {
                "query":          effective_query,
                "papers":         papers,
                "answer":         answer,
                "level":          level,
                "agreements":     agreements,
                "disagreements":  disagreements,
            }
            st.rerun()  # Rerun so hero collapses cleanly

        except ValueError as e:
            _prog.empty()
            st.session_state.search_error = str(e)

# ─── Show error if set ────────────────────────────────────────
if st.session_state.search_error:
    st.error(st.session_state.search_error)

# ═══════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════
if st.session_state.results:
    r             = st.session_state.results
    papers        = r["papers"]
    answer        = r["answer"]
    level         = r["level"]
    agreements    = r["agreements"]
    disagreements = r["disagreements"]
    q             = r["query"]

    # ── Question heading ──────────────────────────────────────
    st.markdown(
        f'<p style="font-family:Inter,sans-serif;font-size:10px;font-weight:500;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:#9B9890;margin:0 0 5px 0;">Question</p>'
        f'<h2 style="font-family:\'DM Serif Display\',serif;font-size:26px;font-weight:400;'
        f'color:#0F0E0C;margin:0 0 22px 0;line-height:1.35;">{q}</h2>',
        unsafe_allow_html=True,
    )

    # ── Stats strip ───────────────────────────────────────────
    contradiction_cell = (
        f'<div style="flex:1;padding:16px 20px;">'
        f'<div style="font-size:9px;letter-spacing:0.16em;text-transform:uppercase;'
        f'color:#9B9890;margin-bottom:8px;font-weight:500;font-family:Inter,sans-serif;">Contradiction</div>'
        f'{level_badge(level)}</div>'
    )
    st.markdown(
        f'<div style="display:flex;background:#FFFFFF;border:1px solid #E0DDD6;'
        f'border-radius:10px;overflow:hidden;margin-bottom:22px;">'
        f'{stat_cell("Sources", len(papers), "of 5,000")}'
        f'{stat_cell("Retrieval", "Dense + BM25", "RRF fusion k=60")}'
        f'{stat_cell("Top RRF", papers[0]["rrf_score"], "best match")}'
        f'{contradiction_cell}'
        f'{stat_cell("Model", "Gemini 2.0", "Flash", border_right=False)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────
    tab_a, tab_s, tab_c = st.tabs([
        "Answer",
        f"Sources  ({len(papers)})",
        f"Contradictions  ·  {level}",
    ])

    # ── ANSWER TAB ────────────────────────────────────────────
    with tab_a:
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;'
            f'padding:28px 32px;font-family:Inter,sans-serif;font-size:15px;font-weight:300;'
            f'line-height:1.85;color:#0F0E0C;margin:16px 0 12px;">{answer}</div>',
            unsafe_allow_html=True,
        )
        # Grounding note
        st.markdown(
            '<div style="background:#F5F2EC;border:1px solid #E0DDD6;border-radius:7px;'
            'padding:10px 14px;font-family:Inter,sans-serif;font-size:12px;color:#9B9890;'
            'line-height:1.5;margin-bottom:18px;">'
            '🔒 Every claim is grounded in retrieved papers. '
            '[1], [2]… refer to sources in the Sources tab. '
            'Conflicts are stated explicitly — never silently blended.</div>',
            unsafe_allow_html=True,
        )
        # Downloads live inside the Answer tab — user has read the answer, now they can save it
        citation_text = "\n".join(
            f"[{i}] {p['metadata']['title']} — arxiv.org/abs/{p['metadata']['arxiv_id']}"
            for i, p in enumerate(papers, 1)
        )
        dl1, dl2, _ = st.columns([1, 1, 4])
        with dl1:
            st.download_button("↓ Save answer", answer, file_name="arxiv_lens_answer.txt", use_container_width=True)
        with dl2:
            st.download_button("↓ Save sources", citation_text, file_name="arxiv_lens_sources.txt", use_container_width=True)

    # ── SOURCES TAB ───────────────────────────────────────────
    with tab_s:
        st.markdown(
            '<p style="font-family:Inter,sans-serif;font-size:13px;color:#9B9890;'
            'margin:16px 0 14px;">Ranked by Reciprocal Rank Fusion — '
            'higher score means stronger agreement between semantic and keyword search.</p>',
            unsafe_allow_html=True,
        )
        for i, p in enumerate(papers, 1):
            aid   = p["metadata"]["arxiv_id"]
            title = p["metadata"]["title"]
            cats  = p["metadata"]["categories"]
            rrf_s = p["rrf_score"]
            abst  = p["document"][:340]
            url   = f"https://arxiv.org/abs/{aid}"

            st.markdown(
                f'<div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;'
                f'padding:18px 22px;margin-bottom:10px;">'
                # Header row
                f'<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:8px;">'
                f'<span style="font-size:10px;font-weight:600;letter-spacing:0.06em;color:#C8B89A;'
                f'background:#F5F2EC;border:1px solid #E0DDD6;padding:3px 7px;border-radius:4px;'
                f'flex-shrink:0;margin-top:2px;font-family:Inter,sans-serif;">[{i}]</span>'
                f'<span style="font-family:\'DM Serif Display\',serif;font-size:16px;color:#0F0E0C;'
                f'line-height:1.35;flex:1;">{title}</span>'
                f'<span style="font-size:10px;color:#9B9890;background:#F5F2EC;border:1px solid #E0DDD6;'
                f'padding:3px 7px;border-radius:4px;flex-shrink:0;margin-top:2px;white-space:nowrap;'
                f'font-family:Inter,sans-serif;">RRF {rrf_s}</span>'
                f'</div>'
                # Meta row
                f'<div style="font-size:11px;color:#9B9890;margin-bottom:10px;font-family:Inter,sans-serif;">'
                f'<a href="{url}" target="_blank" style="color:#6A8CC7;text-decoration:none;">'
                f'arxiv.org/abs/{aid}</a>&nbsp;·&nbsp;{cats}</div>'
                # Abstract
                f'<div style="font-family:Inter,sans-serif;font-size:13px;font-weight:300;color:#6B6860;'
                f'line-height:1.7;border-top:1px solid #E0DDD6;padding-top:12px;">{abst}…'
                f'&nbsp;<a href="{url}" target="_blank" style="color:#6A8CC7;font-size:12px;'
                f'text-decoration:none;white-space:nowrap;">Read full paper →</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── CONTRADICTIONS TAB ────────────────────────────────────
    with tab_c:
        lm = LEVEL_META.get(level, LEVEL_META["UNKNOWN"])
        st.markdown(
            f'<p style="font-family:Inter,sans-serif;font-size:13px;color:#9B9890;margin:16px 0 14px;">'
            f'Gemini cross-analysed all {len(papers)} retrieved papers for agreements and conflicts. '
            f'Contradictions are named explicitly — not hidden or blended away.</p>',
            unsafe_allow_html=True,
        )

        # Level banner
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;padding:12px 16px;'
            f'background:{lm["bg"]};border:1px solid {lm["border"]};border-radius:8px;'
            f'margin-bottom:16px;">'
            f'<span style="font-family:Inter,sans-serif;font-size:10px;font-weight:500;'
            f'color:{lm["color"]};text-transform:uppercase;letter-spacing:0.14em;">Contradiction level</span>'
            f'{level_badge(level)}</div>',
            unsafe_allow_html=True,
        )

        # Agreements
        if agreements:
            st.markdown(
                f'<div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;'
                f'padding:20px 24px;margin-bottom:10px;">'
                f'<p style="font-family:Inter,sans-serif;font-size:10px;font-weight:600;'
                f'letter-spacing:0.14em;text-transform:uppercase;color:#9B9890;margin:0 0 10px;">Agreements</p>'
                f'<div style="font-family:Inter,sans-serif;font-size:14px;font-weight:300;'
                f'line-height:1.8;color:#0F0E0C;">{agreements}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Disagreements
        if disagreements:
            st.markdown(
                f'<div style="background:#FFFFFF;border:1px solid #E0DDD6;border-radius:10px;'
                f'padding:20px 24px;">'
                f'<p style="font-family:Inter,sans-serif;font-size:10px;font-weight:600;'
                f'letter-spacing:0.14em;text-transform:uppercase;color:#9B9890;margin:0 0 10px;">Disagreements</p>'
                f'<div style="font-family:Inter,sans-serif;font-size:14px;font-weight:300;'
                f'line-height:1.8;color:#0F0E0C;">{disagreements}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:40px;padding-top:20px;border-top:1px solid #E0DDD6;">'
        '<p style="font-family:Inter,sans-serif;font-size:11px;color:#B0ADA6;'
        'margin:0;text-align:center;">'
        'ArXiv Lens &nbsp;·&nbsp; BM25 + Dense Hybrid Retrieval &nbsp;·&nbsp; '
        'RRF Fusion &nbsp;·&nbsp; Gemini 2.0 Flash</p></div>',
        unsafe_allow_html=True,
    )