import streamlit as st

from src import config
from src.rag import RAGPipeline

st.set_page_config(page_title="Customer Support Knowledge Bot", page_icon="💬", layout="wide")

@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_pipeline():
    return RAGPipeline()

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}** . page {s['page']}")
            snippet = s["text"][:400] + ("..." if len(s["text"]) > 400 else "")
            st.caption(snippet)


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📚 Knowledge base")
    uploaded = st.file_uploader("Upload PDF files", type="pdf",accept_multiple_files=True)
    if st.button("Add to knowledge base", disabled=not uploaded, use_container_width=True):
        with st.spinner("Reading and indexing documents..."):
            total = sum(pipeline.ingest_pdf(f, f.name) for f in uploaded)
        st.success(f"Indexed {total} chunks from {len(uploaded)} file(s).")

    st.subheader("Indexed Documents")
    sources = pipeline.list_sources()
    if sources:
        for name in sources:
            st.write(f"• {name}")
    else:
        st.caption("No documents yet. Upload a PDF to get started.")
    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col2.button("Delete Docs", use_container_width=True):
        pipeline.reset()
        st.rerun()
    st.caption(f"LLM backend: `{config.LLM_BACKEND}`")

st.title("💬 Customer Support Knowledge Bot")
st.caption("Answers come only from your uploaded documents, with sources shown.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = pipeline.ask(question, history)
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )