# frontend/app.py

import streamlit as st
import requests
import time

# Configuration
API_URL = "http://localhost:8000"

# Page setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 RAG Chatbot - Document Q&A")
st.markdown("Ask questions about your documents and get accurate answers with citations!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Documents to retrieve", 1, 10, 3)
    
    st.divider()
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file and st.button("Process"):
        with st.spinner("Processing..."):
            files = {"file": uploaded_file}
            try:
                response = requests.post(f"{API_URL}/documents/upload", files=files)
                if response.status_code == 200:
                    st.success(f"✅ Processed {response.json()['chunks_created']} chunks")
                else:
                    st.error("❌ Upload failed")
            except:
                st.error("❌ Cannot connect to API")
    
    st.divider()
    st.caption("Built with FastAPI + Streamlit + Ollama")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"{i}. {src['source']} (Page {src['page']}) - Relevance: {src['relevance']:.2f}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": top_k}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", [])
                    })
                    
                    # Show sources
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for i, src in enumerate(data["sources"], 1):
                                st.caption(f"{i}. {src['source']} (Page {src['page']}) - Relevance: {src['relevance']:.2f}")
                else:
                    st.error("❌ Query failed")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Clear chat button
if st.session_state.messages and st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()