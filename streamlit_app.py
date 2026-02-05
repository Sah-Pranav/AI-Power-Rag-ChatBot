# Placeholder
# frontend/app.py

import streamlit as st
import requests
import time

# Configuration
API_URL = "http://localhost:8000"

# Page setup
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
    .example-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-card:hover {
        background-color: #e0e2e6;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = None

# Check API status
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Status
    api_ok = check_api_status()
    if api_ok:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.caption("Make sure FastAPI is running on port 8000")
    
    st.divider()
    
    # Settings
    st.subheader("📊 Query Settings")
    top_k = st.slider("Documents to retrieve", 1, 10, 3, 
                     help="Number of relevant document chunks to retrieve")
    
    st.divider()
    
    # Document Upload
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        if st.button("📥 Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                files = {"file": uploaded_file}
                try:
                    response = requests.post(f"{API_URL}/documents/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Processed {data['chunks_created']} chunks from {data['filename']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Upload failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # Document Info
    st.subheader("📚 Collection Info")
    try:
        response = requests.get(f"{API_URL}/documents/info")
        if response.status_code == 200:
            info = response.json()
            st.metric("Total Documents", info.get('total_documents', 0))
    except:
        st.caption("No data available")
    
    st.divider()
    
    # Clear Chat
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.caption("🚀 Built with FastAPI + Streamlit + Ollama")

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Chatbot</h1>
    <p>Ask questions about your documents and get AI-powered answers with citations</p>
</div>
""", unsafe_allow_html=True)

# Welcome screen (when no messages)
if not st.session_state.messages:
    st.markdown("### 👋 Welcome! Get started by:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**1️⃣ Upload a PDF**\nUse the sidebar to upload your document")
    
    with col2:
        st.info("**2️⃣ Ask Questions**\nType your question in the chat below")
    
    with col3:
        st.info("**3️⃣ Get Answers**\nReceive AI-generated answers with sources")
    
    st.markdown("### 💡 Try these example questions:")
    
    # Example questions
    examples = [
        "What is environment scaffolding?",
        "Who are the authors of this paper?",
        "What are the main contributions?",
        "Summarize the key findings",
    ]
    
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            if st.button(f"💬 {example}", key=f"example_{idx}", use_container_width=True):
                st.session_state.example_clicked = example
                st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 View {len(msg['sources'])} Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    **{i}. {src['source']}** (Page {src['page']})  
                    Relevance: {src['relevance']:.2%}  
                    {src.get('content_preview', '')[:150]}...
                    """)
        if "query_time" in msg:
            st.caption(f"⏱️ Response time: {msg['query_time']:.2f}s")

# Handle example click
if hasattr(st.session_state, 'example_clicked'):
    prompt = st.session_state.example_clicked
    del st.session_state.example_clicked
else:
    prompt = st.chat_input("Ask a question about your documents...")

# Process user input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": top_k},
                    timeout=120
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    query_time = end_time - start_time
                    
                    # Display answer
                    message_placeholder.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📚 View {len(sources)} Sources"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"""
                                **{i}. {src['source']}** (Page {src['page']})  
                                Relevance: {src['relevance']:.2%}  
                                {src.get('content_preview', '')[:150]}...
                                """)
                    
                    st.caption(f"⏱️ Response time: {query_time:.2f}s")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "query_time": query_time
                    })
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The query is taking too long.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


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