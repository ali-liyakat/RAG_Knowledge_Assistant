"""
streamlit_app.py — chat UI for the RAG Knowledge Assistant.
Calls the RAG pipeline directly (no separate API server needed for deployment).
"""
import sys
from pathlib import Path

# Ensure the project root is importable regardless of how/where this script is launched from —
# needed both locally and on Streamlit Cloud, since streamlit run doesn't support python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from pathlib import Path

import streamlit as st

# ---------- Load secrets (Streamlit Cloud) into env vars the pipeline expects ----------
# Locally, python-dotenv (called inside embedder.py/llm.py) still handles .env as before.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        os.environ["QDRANT_URL"] = st.secrets["QDRANT_URL"]
        os.environ["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
except FileNotFoundError:
    pass  # no secrets.toml locally — .env (loaded inside the pipeline modules) is used instead

from src.pipeline import answer_question
from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_and_store

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body { height: 100%; }
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #1A2144 0%, #0A0E27 50%, #05070F 100%);
    min-height: 100vh;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 780px;
}

.main-title {
    font-family: 'Poppins', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.15;
    text-align: center;
    background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 1.08rem;
    margin-top: 0;
    margin-bottom: 2.2rem;
}

.stButton > button {
    width: 100% !important;
    height: 90px !important;
    border-radius: 16px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    background: linear-gradient(135deg, #6366F1, #A855F7) !important;
    color: white !important;
    box-shadow: 0 4px 18px rgba(139, 92, 246, 0.35) !important;
    transition: 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(236, 72, 153, 0.45) !important;
    transform: translateY(-1px);
}
.stButton > button p { color: white !important; font-weight: 700 !important; }

div[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    height: auto !important;
    padding: 0.6rem 1rem !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366F1, #A855F7) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(139, 92, 246, 0.35) !important;
}
div[data-testid="stFormSubmitButton"] button p { color: white !important; }
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 6px 24px rgba(236, 72, 153, 0.45) !important;
}

.stTextInput input {
    background: #10162C !important;
    color: #E5E7EB !important;
    border: 1px solid #2D3452 !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput input::placeholder { color: #6B7280 !important; }

div[data-testid="stChatMessage"] {
    border-radius: 16px;
    border: 1px solid #232A47;
    background: #10162C;
}

.source-chip {
    display: inline-block;
    background: linear-gradient(90deg, #6366F1, #EC4899);
    color: white;
    border-radius: 999px;
    padding: 5px 14px;
    margin: 3px 5px 3px 0;
    font-size: 0.78rem;
    font-weight: 600;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #A78BFA;
    border-radius: 14px;
    padding: 1.2rem;
    background: #10162C;
}
div[data-testid="stFileUploader"] label, div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span { color: #9CA3AF !important; }

h4, .stMarkdown p, .stCaption, p { color: #D1D5DB; }

div[data-testid="stSpinner"] p { color: #A78BFA !important; font-weight: 600; }

.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #232A47;
    color: #6B7280;
    font-size: 0.9rem;
}
.footer b { color: #A78BFA; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<p class="main-title">📚 RAG Knowledge<br>Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions from your ML/DL notes and get grounded, cited answers.</p>',
    unsafe_allow_html=True
)

# ---------- Nav ----------
if "view" not in st.session_state:
    st.session_state.view = "chat"
if "current_qa" not in st.session_state:
    st.session_state.current_qa = None

nav_left, nav_gap, nav_right = st.columns([1, 0.3, 1])
with nav_left:
    if st.button("💬  Ask a Question", use_container_width=True):
        st.session_state.view = "chat"
with nav_right:
    if st.button("📤  Add New Notes", use_container_width=True):
        st.session_state.view = "upload"

st.markdown('<div style="height: 1.8rem"></div>', unsafe_allow_html=True)

# ---------- Chat view ----------
if st.session_state.view == "chat":
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input(
            "Ask", placeholder="Ask something from your notes...", label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and question:
        with st.spinner("🔎 Digging through your notes..."):
            try:
                result = answer_question(question, top_k=8)
                st.session_state.current_qa = {
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                }
            except Exception as e:
                st.session_state.current_qa = {
                    "question": question,
                    "answer": f"Something went wrong: {e}",
                    "sources": [],
                }

    st.markdown('<div style="height: 1.5rem"></div>', unsafe_allow_html=True)

    qa = st.session_state.current_qa
    if qa:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(qa["question"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(qa["answer"])
            if qa["sources"]:
                chips = "".join(
                    f'<span class="source-chip">📄 {s["source"]} · p.{s["page"]}</span>'
                    for s in qa["sources"]
                )
                st.markdown(chips, unsafe_allow_html=True)

# ---------- Upload view ----------
else:
    st.markdown("#### Add a new PDF to your knowledge base")
    st.caption("It'll be chunked, embedded, and made searchable immediately.")

    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("➕ Add to knowledge base", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    raw_pdf_dir = Path("data/raw_pdfs")
                    raw_pdf_dir.mkdir(parents=True, exist_ok=True)
                    save_path = raw_pdf_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    pages = load_pdf(save_path)
                    chunks = chunk_pages(pages)
                    embed_and_store(chunks)

                    st.success(f"Added **{uploaded_file.name}**")
                    c1, c2 = st.columns(2)
                    c1.metric("Pages processed", len(pages))
                    c2.metric("Chunks created", len(chunks))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# ---------- Footer ----------
st.markdown('<div class="footer">Built by <b>Liyakat Ali Joo</b></div>', unsafe_allow_html=True)