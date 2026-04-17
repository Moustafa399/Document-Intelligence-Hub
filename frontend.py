import streamlit as st
import requests
import uuid

# ==============================================================================
# Configuration & Setup
# ==============================================================================
API_BASE_URL = "http://localhost:8007"

st.set_page_config(
    page_title="Document Intelligence Hub", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light/dark responsive CSS for a cleaner look
st.markdown("""
<style>
    /* Improve button appearance */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Improve headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Better chat message spacing */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# Sidebar: Control Panel & Ingestion
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=70)
    st.markdown("## Control Panel")
    st.caption(f"Active Session: `{st.session_state.session_id}`")
    st.divider()
    
    st.markdown("### 📥 Document Ingestion")
    author_name = st.text_input("Author Name", value="System User", help="Tag the uploaded documents with this author name.")
    
    uploaded_files = st.file_uploader(
        "Upload your files here", 
        accept_multiple_files=True,
        type=["pdf", "docx", "xlsx", "csv", "pptx", "png", "jpg"],
        help="Supported formats: PDF, Word, Excel, CSV, PPT, Images."
    )
    
    if st.button("🚀 Upload & Process Files", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("⚠️ Please select at least one file first.")
        else:
            with st.status("Processing Documents...", expanded=True) as status:
                try:
                    st.write("Preparing files for transfer...")
                    files_data = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                    data = {"author": author_name, "session_id": st.session_state.session_id}
                    
                    st.write("Sending to backend for OCR and Embedding...")
                    response = requests.post(f"{API_BASE_URL}/extract", files=files_data, data=data, timeout=180)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        status.update(label=f"✅ Successfully processed {res_json['processed_count']} files!", state="complete", expanded=False)
                        st.toast("Files successfully added to your Knowledge Base!", icon="🎉")
                        
                        # Show details in an expander
                        with st.expander("📄 View Processing Logs"):
                            for doc in res_json["documents"]:
                                if doc["status"] == "success":
                                    st.success(f"**{doc['filename']}** - Processed")
                                else:
                                    st.error(f"**{doc['filename']}** - {doc.get('error', 'Unknown Error')}")
                    else:
                        status.update(label="❌ Server Error", state="error", expanded=True)
                        st.error(f"Backend returned: {response.status_code}\n{response.text}")
                except requests.exceptions.ConnectionError:
                    status.update(label="❌ Connection Failed", state="error", expanded=True)
                    st.error("Could not connect to the FastAPI server on port 8007.")
                except Exception as e:
                    status.update(label="❌ Unexpected Error", state="error", expanded=True)
                    st.error(str(e))

    st.divider()
    st.markdown("### ⚙️ Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# Main Layout: AI Chat Interface
# ==============================================================================
st.markdown('<div class="main-header">📚 Document Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions and get answers instantly based on your uploaded documents.</div>', unsafe_allow_html=True)

# Display a welcome message if chat is empty
if not st.session_state.messages:
    st.info("👋 **Welcome!** Upload some documents from the sidebar, then ask me anything about them.")

# Display chat history
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask about your documents (e.g., 'Summarize the financial report')..."):
    
    # 1. Show user message
    st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Process Assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner("Analyzing documents..."):
            try:
                chat_data = {"query": prompt, "top_k": 5, "session_id": st.session_state.session_id}
                response = requests.post(f"{API_BASE_URL}/chat", json=chat_data, timeout=60)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer received.")
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"❌ Server Error: {response.status_code}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to the server (port 8007)."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
