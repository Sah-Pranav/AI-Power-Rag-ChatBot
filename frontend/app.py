import os
import time
import requests
import streamlit as st
from urllib.parse import quote

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Helpers ----------------
def safe_request(method: str, url: str, timeout: int = 20, **kwargs):
    """Safe wrapper around requests.request"""
    try:
        return requests.request(method, url, timeout=timeout, **kwargs)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def check_api_status() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_ok = check_api_status()
    if api_ok:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.caption("Make sure FastAPI is running on port 8000")

    st.divider()

    st.subheader("📊 Query Settings")
    top_k = st.slider("Documents to retrieve", 1, 10, 3)

    st.divider()

    # Upload
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and st.button("📥 Process Document", use_container_width=True):
        with st.spinner("Processing document..."):
            resp = safe_request(
                "POST",
                f"{API_URL}/documents/upload",
                files={"file": uploaded_file},
                timeout=120,
            )
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ Processed {data['chunks_created']} chunks from {data['filename']}")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"❌ Upload failed: {resp.text if resp else ''}")

    st.divider()

    # Document Manager
    st.subheader("📚 Document Manager")
    if st.button("🔄 Refresh document list", use_container_width=True):
        st.rerun()

    resp = safe_request("GET", f"{API_URL}/documents/list", timeout=20)
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        documents = data.get("documents", [])
        total_docs = data.get("total_documents", 0)

        st.caption(f"Total chunks in DB: {total_docs}")

        if not documents:
            st.info("No documents found in the collection yet.")
        else:
            for doc in documents:
                name = doc.get("name", doc.get("source", "Unknown"))
                chunk_count = doc.get("chunk_count", 0)

                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(f"Chunks: {chunk_count}")

                    if st.button(f"🗑️ Delete {name}", key=f"del_{name}", use_container_width=True):
                        del_resp = safe_request(
                            "DELETE",
                            f"{API_URL}/documents/source/{quote(name)}",
                            timeout=60,
                        )
                        if del_resp is not None and del_resp.status_code == 200:
                            st.success(f"Deleted: {name}")
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {del_resp.text if del_resp else ''}")
    else:
        st.warning("Could not load document list. Is the API running?")

    st.divider()

    # Danger zone
    st.subheader("⚠️ Danger Zone")
    confirm_clear = st.checkbox("I understand this will delete all documents", value=False)

    if st.button("🔥 Clear ALL documents", use_container_width=True, disabled=not confirm_clear):
        clear_resp = safe_request("DELETE", f"{API_URL}/documents/all", timeout=60)
        if clear_resp is not None and clear_resp.status_code == 200:
            st.success("All documents cleared.")
            st.rerun()
        else:
            st.error(f"Clear failed: {clear_resp.text if clear_resp else ''}")

    st.divider()

    # Clear Chat (ONLY show if chat exists)
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

    # Developer info (always visible)
    st.subheader("👨‍💻 Developed by")
    st.markdown("**Pranav Sah**")
    st.caption("🚀 Built with FastAPI + Streamlit + Ollama")


# ---------------- Main ----------------
st.markdown(
    """
<div class="main-header">
    <h1>🤖 RAG Chatbot</h1>
    <p>Ask questions about your documents and get AI-powered answers with citations</p>
</div>
""",
    unsafe_allow_html=True,
)

# Welcome screen (ONLY when no messages)
if not st.session_state.messages:
    st.markdown("### 👋 Welcome! Get started by:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1️⃣ Upload a PDF**\nUse the sidebar to upload your document")
    with col2:
        st.info("**2️⃣ Ask Questions**\nType your question in the chat below")
    with col3:
        st.info("**3️⃣ Get Answers**\nReceive AI-generated answers with sources")
    st.divider()


# ---------------- Chat History ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📚 View {len(msg['sources'])} Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src.get('source', 'Unknown')}** (Page {src.get('page', 'N/A')})")
                    rel = src.get("relevance")
                    if rel is not None:
                        st.caption(f"Relevance: {rel:.3f}")
                    st.code(src.get("content_preview", ""), language="text")

        if msg.get("query_time") is not None:
            st.caption(f"⏱️ Response time: {msg['query_time']:.2f}s")


# ---------------- Chat Input ----------------
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            resp = safe_request(
                "POST",
                f"{API_URL}/query",
                json={"question": prompt, "top_k": top_k},
                timeout=120,
            )
            query_time = time.time() - start_time

            if resp is not None and resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 View {len(sources)} Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}. {src.get('source', 'Unknown')}** (Page {src.get('page', 'N/A')})")
                            rel = src.get("relevance")
                            if rel is not None:
                                st.caption(f"Relevance: {rel:.3f}")
                            st.code(src.get("content_preview", ""), language="text")

                st.caption(f"⏱️ Response time: {query_time:.2f}s")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources, "query_time": query_time}
                )
            else:
                st.error(f"❌ Error: {resp.text if resp else 'No response'}")
