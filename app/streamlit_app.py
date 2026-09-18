"""
streamlit_app.py — chat UI for the RAG Knowledge Assistant.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="centered")
st.title("📚 RAG Knowledge Assistant")
st.caption("Ask questions grounded in your own ML/DL notes.")

tab_chat, tab_upload = st.tabs(["💬 Ask a question", "📤 Add new notes"])

# --- Chat tab ---
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- {s['source']}, page {s['page']} (score: {s['score']})")

    question = st.chat_input("Ask something from your notes...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question, "top_k": 5},
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                st.write(f"- {s['source']}, page {s['page']} (score: {s['score']})")

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources
                    })
                except requests.exceptions.RequestException as e:
                    error_msg = f"Couldn't reach the API: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

# --- Upload tab ---
with tab_upload:
    st.write("Add new PDF notes to the knowledge base.")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Add to knowledge base"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                    response.raise_for_status()
                    result = response.json()

                    if result.get("status") == "success":
                        st.success(
                            f"Added **{result['filename']}** — "
                            f"{result['pages_processed']} pages, {result['chunks_created']} chunks."
                        )
                    else:
                        st.error(result.get("error", "Something went wrong."))
                except requests.exceptions.RequestException as e:
                    st.error(f"Couldn't reach the API: {e}")