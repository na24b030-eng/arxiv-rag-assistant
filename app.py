import re
import pickle
from collections import defaultdict

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from google import genai

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ArXiv Lens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULT_STATES = {
    "history": [],
    "last_query": "",
    "current_query": "",
    "trigger_search": False,
    "results": None,
}

for key, value in DEFAULT_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# DESIGN SYSTEM
# --------------------------------------------------

st.markdown(
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    font-family: Inter, sans-serif;
    background:#ffffff;
    color:#111827;
}

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

.block-container{
    max-width:1100px;
    padding-top:1rem;
    padding-bottom:4rem;
}

/* -----------------------
NAVBAR
----------------------- */

.navbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:40px;
}

.logo{
    font-size:24px;
    font-weight:700;
    color:#111827;
}

/* -----------------------
HERO
----------------------- */

.hero-title{
    font-size:56px;
    font-weight:700;
    text-align:center;
    color:#111827;
    margin-top:30px;
}

.hero-subtitle{
    text-align:center;
    font-size:18px;
    color:#6B7280;
    max-width:700px;
    margin:auto;
    margin-top:16px;
    margin-bottom:40px;
}

/* -----------------------
SEARCH
----------------------- */

.search-card{
    border:1px solid #E5E7EB;
    border-radius:20px;
    padding:24px;
    background:white;
}

.stTextInput input{
    height:64px !important;
    font-size:18px !important;
    border-radius:16px !important;
}

.stButton button{
    border-radius:12px !important;
    font-weight:600 !important;
}

/* -----------------------
CARDS
----------------------- */

.metric-card{
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:18px;
    background:white;
}

.section-title{
    font-size:20px;
    font-weight:600;
    margin-top:30px;
    margin-bottom:15px;
}

/* -----------------------
HISTORY
----------------------- */

.history-label{
    color:#6B7280;
    text-transform:uppercase;
    letter-spacing:1px;
    font-size:12px;
    margin-bottom:12px;
}

/* -----------------------
SIDEBAR
----------------------- */

[data-testid="stSidebar"]{
    background:#F9FAFB;
}

</style>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# TOP NAVIGATION
# --------------------------------------------------

left, right = st.columns([10,1])

with left:
    st.markdown(
        "<div class='logo'>🔬 ArXiv Lens</div>",
        unsafe_allow_html=True
    )

with right:
    st.button("⚙", key="settings_btn")

# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
"""
<div class='hero-title'>
Ask ML Research Questions
</div>

<div class='hero-subtitle'>
Search 5,000+ ArXiv papers using hybrid retrieval,
grounded generation and contradiction analysis.
</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# SEARCH CARD
# --------------------------------------------------

st.markdown(
"<div class='search-card'>",
unsafe_allow_html=True
)

query = st.text_input(
    "",
    value=st.session_state.current_query,
    placeholder="Ask a question about machine learning research..."
)

search_clicked = st.button(
    "Search",
    type="primary",
    use_container_width=True
)

st.markdown(
"</div>",
unsafe_allow_html=True
)

# --------------------------------------------------
# EXAMPLES
# --------------------------------------------------

st.markdown(
"<div class='section-title'>Popular Searches</div>",
unsafe_allow_html=True
)

EXAMPLES = [
    "What is transfer learning?",
    "How do transformers scale?",
    "What is RLHF?",
    "How does BERT work?",
    "What are attention mechanisms?",
    "What is federated learning?"
]

cols = st.columns(3)

for idx, example in enumerate(EXAMPLES):

    with cols[idx % 3]:

        if st.button(
            example,
            key=f"example_{idx}",
            use_container_width=True
        ):
            st.session_state.current_query = example
            st.session_state.trigger_search = True
            st.rerun()

# --------------------------------------------------
# HISTORY
# --------------------------------------------------

if st.session_state.history:

    st.markdown("---")

    st.markdown(
        "<div class='history-label'>Recent Searches</div>",
        unsafe_allow_html=True
    )

    history_cols = st.columns(4)

    recent = st.session_state.history[-8:]

    for idx, item in enumerate(reversed(recent)):

        with history_cols[idx % 4]:

            if st.button(
                item,
                key=f"history_{idx}",
                use_container_width=True
            ):
                st.session_state.current_query = item
                st.session_state.trigger_search = True
                st.rerun()

# --------------------------------------------------
# SEARCH LOGIC
# --------------------------------------------------

should_run = False
effective_query = query

if search_clicked and query:

    should_run = True
    effective_query = query

elif st.session_state.trigger_search:

    should_run = True

    effective_query = st.session_state.current_query

    st.session_state.trigger_search = False

# --------------------------------------------------
# PART 2 ENTRY POINT
# --------------------------------------------------

if should_run:

    st.session_state.last_query = effective_query

    if effective_query not in st.session_state.history:
        st.session_state.history.append(effective_query)

    st.info(
        f"Searching: {effective_query}"
    )

    # ==================================================
# CONFIG
# ==================================================

EMBEDDINGS_PATH = "embeddings.npy"
CHUNKS_PATH = "chunks.pkl"
BM25_PATH = "bm25_index.pkl"

TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_FINAL = 5
RRF_K = 60

MODEL = "gemini-3.5-flash"

# ==================================================
# PROMPTS
# ==================================================

RAG_SYSTEM_PROMPT = """
You are a precise ML research assistant.

Rules:

1. Use ONLY the provided context.
2. Cite every factual claim inline as [N].
3. Never invent information.
4. If information is missing, say:
   "Not covered in the retrieved papers."
5. Maximum 300 words.
6. End with:

### Sources
"""

CONTRADICTION_PROMPT = """
You are comparing ML papers.

Query:
{query}

Context:
{context}

Return:

1. CONTRADICTION LEVEL:
NONE / LOW / MODERATE / HIGH

2. AGREEMENTS

3. DISAGREEMENTS

Reference papers using [N].
"""

# ==================================================
# RESOURCE LOADING
# ==================================================

@st.cache_resource(show_spinner=False)
def load_embed_model():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_embeddings():

    return np.load(
        EMBEDDINGS_PATH
    )


@st.cache_resource(show_spinner=False)
def load_chunks():

    with open(CHUNKS_PATH, "rb") as f:

        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_bm25():

    with open(BM25_PATH, "rb") as f:

        return pickle.load(f)

# ==================================================
# LOAD RESOURCES
# ==================================================

embed_model = load_embed_model()

embeddings = load_embeddings()

chunks = load_chunks()

bm25_index = load_bm25()

# ==================================================
# TOKENIZATION
# ==================================================

def tokenise(text):

    return re.sub(
        r"[^\w\s]",
        "",
        text.lower()
    ).split()

# ==================================================
# DENSE RETRIEVAL
# ==================================================

def dense_retrieve(
    query,
    embed_model,
    embeddings,
    chunks,
    top_k=TOP_K_DENSE
):

    scores = cosine_similarity(
        embed_model.encode([query]),
        embeddings
    )[0]

    top_idx = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for i in top_idx:

        results.append(
            {
                "chunk_id":
                    chunks[i]["chunk_id"],

                "score":
                    float(scores[i]),

                "document":
                    chunks[i]["chunk_text"],

                "metadata":
                    {
                        k: chunks[i][k]
                        for k in
                        (
                            "arxiv_id",
                            "title",
                            "categories"
                        )
                    }
            }
        )

    return results

# ==================================================
# SPARSE RETRIEVAL
# ==================================================

def sparse_retrieve(
    query,
    bm25_index,
    chunks,
    top_k=TOP_K_SPARSE
):

    scores = bm25_index.get_scores(
        tokenise(query)
    )

    top_idx = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for i in top_idx:

        results.append(
            {
                "chunk_id":
                    chunks[i]["chunk_id"],

                "score":
                    float(scores[i]),

                "document":
                    chunks[i]["chunk_text"],

                "metadata":
                    {
                        k: chunks[i][k]
                        for k in
                        (
                            "arxiv_id",
                            "title",
                            "categories"
                        )
                    }
            }
        )

    return results

# ==================================================
# RRF FUSION
# ==================================================

def rrf_merge(
    dense,
    sparse,
    k=RRF_K,
    top_k=TOP_K_FINAL
):

    scores = defaultdict(float)

    chunk_map = {}

    for rank, hit in enumerate(
        dense,
        start=1
    ):

        scores[
            hit["chunk_id"]
        ] += 1 / (k + rank)

        chunk_map[
            hit["chunk_id"]
        ] = hit

    for rank, hit in enumerate(
        sparse,
        start=1
    ):

        scores[
            hit["chunk_id"]
        ] += 1 / (k + rank)

        if hit["chunk_id"] not in chunk_map:

            chunk_map[
                hit["chunk_id"]
            ] = hit

    final_results = []

    ranked = sorted(
        scores.items(),
        key=lambda x: -x[1]
    )[:top_k]

    for cid, score in ranked:

        item = dict(
            chunk_map[cid]
        )

        item["rrf_score"] = round(
            score,
            6
        )

        final_results.append(
            item
        )

    return final_results

# ==================================================
# HYBRID RETRIEVAL
# ==================================================

def hybrid_retrieve(
    query,
    embed_model,
    embeddings,
    bm25_index,
    chunks
):

    dense = dense_retrieve(
        query,
        embed_model,
        embeddings,
        chunks
    )

    sparse = sparse_retrieve(
        query,
        bm25_index,
        chunks
    )

    return rrf_merge(
        dense,
        sparse
    )

# ==================================================
# CONTEXT BUILDING
# ==================================================

def build_context(
    papers
):

    context = []

    for i, paper in enumerate(
        papers,
        start=1
    ):

        context.append(
            f"""
[{i}]
Title: {paper['metadata']['title']}
ArXiv: {paper['metadata']['arxiv_id']}

Text:
{paper['document']}
"""
        )

    return "\n\n".join(
        context
    )

# ==================================================
# GEMINI
# ==================================================

def ask_gemini(
    prompt,
    client
):

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Error: {e}"


def get_answer(
    query,
    papers,
    client
):

    context = build_context(
        papers
    )

    prompt = f"""
{RAG_SYSTEM_PROMPT}

CONTEXT:

{context}

QUESTION:

{query}

ANSWER:
"""

    return ask_gemini(
        prompt,
        client
    )


def get_contradiction(
    query,
    papers,
    client
):

    return ask_gemini(

        CONTRADICTION_PROMPT.format(
            query=query,
            context=build_context(
                papers
            )
        ),

        client
    )

# ==================================================
# EXECUTE SEARCH
# ==================================================

if should_run:

    if not api_key:

        st.error(
            "Please enter your Gemini API key."
        )

        st.stop()

    with st.spinner(
        "Searching research papers..."
    ):

        client = genai.Client(
            api_key=api_key
        )

        papers = hybrid_retrieve(
            effective_query,
            embed_model,
            embeddings,
            bm25_index,
            chunks
        )

        answer = get_answer(
            effective_query,
            papers,
            client
        )

        contradiction_report = (
            get_contradiction(
                effective_query,
                papers,
                client
            )
        )

    st.session_state.results = {

        "query":
            effective_query,

        "papers":
            papers,

        "answer":
            answer,

        "contradictions":
            contradiction_report
    }

    # ==================================================
# RESULTS UI
# ==================================================

if st.session_state.results:

    result = st.session_state.results

    papers = result["papers"]
    answer = result["answer"]
    contradictions = result["contradictions"]
    query = result["query"]

    st.markdown("---")

    # ==========================================
    # QUERY HEADER
    # ==========================================

    st.markdown(
        f"""
        <div style="
            font-size:14px;
            color:#6B7280;
            margin-bottom:8px;
        ">
        QUESTION
        </div>

        <div style="
            font-size:30px;
            font-weight:700;
            margin-bottom:25px;
            color:#111827;
        ">
        {query}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # METRICS
    # ==========================================

    m1, m2, m3 = st.columns(3)

    with m1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:12px;color:#6B7280;">
                SOURCES USED
                </div>

                <div style="
                    font-size:28px;
                    font-weight:700;
                    margin-top:8px;
                ">
                {len(papers)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:

        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size:12px;color:#6B7280;">
                RETRIEVAL
                </div>

                <div style="
                    font-size:20px;
                    font-weight:700;
                    margin-top:8px;
                ">
                Hybrid
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:

        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size:12px;color:#6B7280;">
                MODEL
                </div>

                <div style="
                    font-size:20px;
                    font-weight:700;
                    margin-top:8px;
                ">
                Gemini
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # ACTIONS
    # ==========================================

    a1, a2, a3 = st.columns([1,1,5])

    with a1:

        st.download_button(
            "📄 Answer",
            answer,
            file_name="answer.txt",
            use_container_width=True
        )

    with a2:

        citation_text = ""

        for idx, paper in enumerate(
            papers,
            start=1
        ):

            citation_text += (
                f"[{idx}] "
                f"{paper['metadata']['title']}\n"
            )

        st.download_button(
            "📚 Sources",
            citation_text,
            file_name="sources.txt",
            use_container_width=True
        )

    # ==========================================
    # ANSWER CARD
    # ==========================================

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:700;
            margin-bottom:15px;
            margin-top:10px;
        ">
        Answer
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            border:1px solid #E5E7EB;
            border-radius:18px;
            padding:25px;
            background:white;
            line-height:1.9;
            font-size:16px;
        ">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # SOURCES
    # ==========================================

    with st.expander(
        f"📚 Sources Used ({len(papers)})",
        expanded=False
    ):

        for idx, paper in enumerate(
            papers,
            start=1
        ):

            title = paper["metadata"]["title"]

            arxiv_id = (
                paper["metadata"]["arxiv_id"]
            )

            categories = (
                paper["metadata"]["categories"]
            )

            snippet = paper["document"][:350]

            url = (
                f"https://arxiv.org/abs/{arxiv_id}"
            )

            st.markdown(
                f"""
                <div style="
                    border:1px solid #E5E7EB;
                    border-radius:16px;
                    padding:20px;
                    margin-bottom:15px;
                    background:white;
                ">

                <div style="
                    font-size:18px;
                    font-weight:600;
                    margin-bottom:8px;
                ">
                [{idx}] {title}
                </div>

                <div style="
                    font-size:13px;
                    color:#6B7280;
                    margin-bottom:12px;
                ">
                {categories}
                </div>

                <div style="
                    color:#374151;
                    line-height:1.7;
                ">
                {snippet}...
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.link_button(
                "Open Paper",
                url,
                use_container_width=False
            )

    # ==========================================
    # CONTRADICTIONS
    # ==========================================

    with st.expander(
        "⚠ Contradiction Analysis",
        expanded=False
    ):

        text_upper = contradictions.upper()

        if "HIGH" in text_upper:

            badge = """
            <span style="
                background:#FEE2E2;
                color:#991B1B;
                padding:6px 12px;
                border-radius:999px;
                font-weight:600;
            ">
            HIGH
            </span>
            """

        elif "MODERATE" in text_upper:

            badge = """
            <span style="
                background:#FEF3C7;
                color:#92400E;
                padding:6px 12px;
                border-radius:999px;
                font-weight:600;
            ">
            MODERATE
            </span>
            """

        elif "LOW" in text_upper:

            badge = """
            <span style="
                background:#DBEAFE;
                color:#1E40AF;
                padding:6px 12px;
                border-radius:999px;
                font-weight:600;
            ">
            LOW
            </span>
            """

        else:

            badge = """
            <span style="
                background:#DCFCE7;
                color:#166534;
                padding:6px 12px;
                border-radius:999px;
                font-weight:600;
            ">
            NONE
            </span>
            """

        st.markdown(
            badge,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="
                border:1px solid #E5E7EB;
                border-radius:16px;
                padding:20px;
                background:white;
                line-height:1.8;
            ">
            {contradictions}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================================
    # FOOTER
    # ==========================================

    st.markdown("---")

    st.caption(
        "ArXiv Lens • Hybrid Retrieval (BM25 + Embeddings) • RRF Fusion • Gemini Grounded Answers"
    )