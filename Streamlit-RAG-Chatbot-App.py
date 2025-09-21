"""
Streamlit RAG Chatbot — Multi-user, Persistent, Optimized (Mistral)
------------------------------------------------------------------
Features:
- ChatGPT-style chat with green (user) and white (bot) bubbles
- Dynamic bubble sizing
- Centered chat & input bar
- Hidden expandable chat history
- New Chat button
- Multi-user support with persistent Chroma index & chat history

This file includes a robust `clear_user_index` implementation that:
- attempts multiple safe shutdown methods on Chroma objects
- forces garbage collection and retries deletion
- if immediate deletion fails, renames the directory and starts a background thread to retry deletion
- as a last resort on Windows, schedules deletion on next reboot

Notes:
- The approach prefers graceful shutdown, then retries; if the OS keeps a handle open the code will try safe fallbacks.
- Background retry thread is daemonized so it won't block Streamlit shutdown.

"""

import os
import json
import shutil
import tempfile
import threading
import time
import gc
import uuid
import ctypes
import stat
from typing import List
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------- Configuration ---------------------------
BASE_USERS_DIR = os.path.join(os.getcwd(), "users")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "mistral"  # ⚡ faster model
CHAT_HISTORY_FILE_TEMPLATE = "chat_history_{username}.json"

os.makedirs(BASE_USERS_DIR, exist_ok=True)

# --------------------------- CSS Styling ---------------------------
st.markdown("""
<style>
/* Center chat container */
.chat-wrapper {
    display: flex;
    justify-content: center;
    width: 100%;
    padding: 20px 50px;
    box-sizing: border-box;
}
.chat-container {
    max-width: 700px;
    width: 100%;
}

/* Chat bubbles */
.user-msg, .bot-msg {
    font-size: 14px;
    padding: 10px 14px;
    border-radius: 15px;
    margin-bottom: 8px;
    display: inline-block;
    max-width: 50%;   /* narrower like ChatGPT */
    word-wrap: break-word;
}

.user-msg {
    background-color: #DCF8C6; /* green */
    float: right;
    clear: both;
    text-align: right;
}
.bot-msg {
    background-color: #F1F0F0; /* white */
    float: left;
    clear: both;
    text-align: left;
}

.chat-scroll {
    max-height: 500px;
    overflow-y: auto;
    padding-bottom: 10px;
}
.chat-container::after {
    content: "";
    display: table;
    clear: both;
}

/* Input bar like ChatGPT */
.stChatInput {
    display: flex !important;
    justify-content: center;  /* keeps the input bar centered */
    position: relative;        /* ensures send icon stays at right */
    width: 100%;
}

.stChatInput textarea {
    max-width: 500px;
    height: 38px;
    min-height: 38px;
    margin: 10px 0;
    display: block;
    border-radius: 18px;
    padding: 10px 50px 10px 16px; /* extra right padding for send icon */
    font-size: 15px;
    line-height: 1.4;
    resize: vertical;
    text-align: left;
    white-space: pre-wrap;
}


/* Sidebar expander font size */
.sidebar .stExpanderHeader { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --------------------------- Helpers ---------------------------

def user_dir(username: str) -> str:
    return os.path.join(BASE_USERS_DIR, username)


def user_persist_dir(username: str) -> str:
    return os.path.join(user_dir(username), "chroma_persist")


def chat_history_path(username: str) -> str:
    return os.path.join(user_dir(username), CHAT_HISTORY_FILE_TEMPLATE.format(username=username))


def ensure_user_dirs(username: str):
    os.makedirs(user_persist_dir(username), exist_ok=True)


def load_pdfs_to_docs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List:
    docs = []
    for uploaded in uploaded_files:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        try:
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source_file"] = uploaded.name
            docs.extend(file_docs)
        except Exception as e:
            st.error(f"Error loading {uploaded.name}: {e}")
    return docs


def split_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def create_or_load_vectorstore(docs, embeddings, persist_directory: str, reindex: bool = False):
    os.makedirs(persist_directory, exist_ok=True)
    if reindex and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    if docs:  # always add new docs
        db.add_documents(docs)
        db.persist()
        return db, True

    return db, False



def build_conversational_chain(db, llm_model_name: str, top_k: int):
    llm = OllamaLLM(model=llm_model_name, temperature=0, num_ctx=2048, timeout=20)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain


def save_chat_history(username: str, history: list):
    ensure_user_dirs(username)
    path = chat_history_path(username)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_chat_history(username: str) -> list:
    path = chat_history_path(username)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def clear_chat_history(username: str):
    path = chat_history_path(username)
    if os.path.exists(path):
        os.remove(path)


# --------------------------- Robust clear_user_index ---------------------------

def _handle_remove_error(func, path, exc_info):
    """onerror handler for shutil.rmtree: try to make file writable and retry once."""
    try:
        # make it writable
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        # swallow: caller will handle final failure
        pass


def try_close_chroma(obj):
    """Try multiple strategies to close/teardown a Chroma DB or client-like object."""
    if obj is None:
        return
    # Common attrs to try on the DB object itself
    try_methods = [
        'teardown', 'shutdown', 'close', 'persist', 'reset', 'stop'
    ]
    for m in try_methods:
        try:
            if hasattr(obj, m):
                try:
                    getattr(obj, m)()
                except Exception:
                    pass
        except Exception:
            pass

    # Try underlying client
    client = getattr(obj, '_client', None) or getattr(obj, 'client', None) or getattr(obj, '_chroma_client', None)
    if client is not None:
        for m in try_methods:
            try:
                if hasattr(client, m):
                    try:
                        getattr(client, m)()
                    except Exception:
                        pass
            except Exception:
                pass

    # Try to close collection if available
    coll = getattr(obj, '_collection', None) or getattr(obj, 'collection', None)
    if coll is not None:
        try:
            if hasattr(coll, 'close'):
                try:
                    coll.close()
                except Exception:
                    pass
        except Exception:
            pass


def schedule_delete_on_reboot(path: str):
    """Schedule directory deletion on next reboot (Windows only)."""
    if os.name != 'nt':
        raise NotImplementedError("schedule_delete_on_reboot is Windows-only")
    MOVEFILE_DELAY_UNTIL_REBOOT = 0x00000004
    res = ctypes.windll.kernel32.MoveFileExW(ctypes.c_wchar_p(path), None, MOVEFILE_DELAY_UNTIL_REBOOT)
    if res == 0:
        raise OSError("MoveFileExW failed to schedule deletion on reboot")


def clear_user_index(username: str, background_retry_seconds: int = 60):
    """Safely clear a user's Chroma index directory.

    Strategy:
    1. Close any DB/chain objects stored in st.session_state and globals.
    2. Force GC and try immediate rmtree with an onerror handler.
    3. Retry a few times with increasing sleeps.
    4. If deletion still fails, rename the folder and start a daemon thread that keeps retrying deletion for `background_retry_seconds` seconds.
    5. As last resort on Windows, schedule deletion on reboot.
    """
    p = user_persist_dir(username)
    if not os.path.exists(p):
        return

    # 1) Close DB and chain from session state if present
    db_key = f"db_{username}"
    chain_key = f"chain_{username}"

    for key in (db_key, chain_key):
        if key in st.session_state:
            obj = st.session_state.pop(key, None)
            try_close_chroma(obj)

    # Also try globals if you (or other code) left a reference
    g_db = globals().get('db')
    if g_db is not None:
        try_close_chroma(g_db)
        globals().pop('db', None)

    gc.collect()

    last_exc = None
    # 2) Try immediate deletion with retries
    for attempt in range(6):
        try:
            shutil.rmtree(p, onerror=_handle_remove_error)
            return
        except PermissionError as e:
            last_exc = e
            # backoff a bit, hoping that OS releases locks
            time.sleep(0.5 * (attempt + 1))
            gc.collect()
        except FileNotFoundError:
            return
        except Exception as e:
            last_exc = e
            break

    # 3) If still present, attempt to atomically rename to a _to_delete_ dir
    tmp = None
    try:
        tmp = p + "_to_delete_" + uuid.uuid4().hex
        os.rename(p, tmp)
    except Exception:
        tmp = None

    if tmp:
        # 4) start a background thread to retry deletion for a while
        def _bg_delete(path, timeout_secs):
            deadline = time.time() + timeout_secs
            while time.time() < deadline:
                try:
                    shutil.rmtree(path, onerror=_handle_remove_error)
                    return
                except Exception:
                    time.sleep(1)
            # last resort: schedule on reboot (Windows-only)
            if os.name == 'nt':
                try:
                    schedule_delete_on_reboot(path)
                except Exception:
                    pass

        t = threading.Thread(target=_bg_delete, args=(tmp, background_retry_seconds), daemon=True)
        t.start()
        return

    # 5) If rename failed, try to schedule deletion on reboot (Windows only)
    if os.name == 'nt':
        try:
            schedule_delete_on_reboot(p)
            return
        except Exception:
            pass

    # If we reach here, raise last exception so the caller can see what went wrong
    raise last_exc or RuntimeError("Failed to clear user index")


# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="RAG PDF Chatbot — Multi-user", layout="wide")

with st.sidebar:
    st.title("Login & Index")
    username = st.text_input("Username", value="alice")
    st.markdown("---")

    # --- New Chat button at top of sidebar ---
    if st.sidebar.button("New Chat", use_container_width=True):
        if username:
            clear_chat_history(username)
            st.session_state.pop(f"history_{username}", None)
            st.success("Started a new chat!")


    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files to index", type=["pdf"], accept_multiple_files=True)

    with st.expander("⚙️ Advanced (optional)"):
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=100)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=50)
        top_k = st.number_input("Retriever top_k", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        embed_model = st.text_input("Embedding model (Ollama)", value=DEFAULT_EMBED_MODEL)
        llm_model = st.text_input("LLM model (Ollama)", value=DEFAULT_LLM_MODEL)
        reindex = st.checkbox("Force re-index (delete previous index)")

    if st.button("Process & Index PDFs", key="process_index"):
        if not username:
            st.warning("Please enter a username before indexing.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF to index.")
        else:
            with st.spinner("Indexing PDFs — one-time cost per user..."):
                ensure_user_dirs(username)
                docs = load_pdfs_to_docs(uploaded_files)
                if not docs:
                    st.error("No text could be extracted from the uploaded PDFs.")
                else:
                    split = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    embeddings = OllamaEmbeddings(model=embed_model)
                    persist_dir = user_persist_dir(username)
                    db, created = create_or_load_vectorstore(split, embeddings, persist_dir, reindex=reindex)

                    st.session_state[f"db_{username}"] = db
                    st.session_state[f"chain_{username}"] = build_conversational_chain(db, llm_model, int(top_k))
                    st.success("Index ready. Future queries will be fast.")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        if username:
            clear_chat_history(username)
            st.session_state.pop(f"history_{username}", None)
            st.success("Chat history cleared for user.")

    if st.button("Clear Index", use_container_width=True):
        if username:
            try:
                clear_user_index(username)
                # ensure session_state references removed
                st.session_state.pop(f"db_{username}", None)
                st.session_state.pop(f"chain_{username}", None)
                st.success("Index cleared for user.")
            except Exception as e:
                st.error(f"Failed to clear index: {e}")


# --------------------------- Main Chat ---------------------------
st.title(f"{username}'s Chatbot")
if not username:
    st.warning("Enter a username in the sidebar to continue.")
    st.stop()

if f"history_{username}" not in st.session_state:
    st.session_state[f"history_{username}"] = load_chat_history(username)

chain_key = f"chain_{username}"
history = st.session_state[f"history_{username}"]

# Display chat in centered container
st.markdown('<div class="chat-wrapper"><div class="chat-container chat-scroll">', unsafe_allow_html=True)
for turn in history[-50:]:
    role = turn["role"]
    content = turn["content"]
    if role == "user":
        st.markdown(f'<div class="user-msg">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{content}</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# Hidden expandable chat history
# --- Sidebar Chat History ---
with st.sidebar.expander("Chat History", expanded=False):
    if history:
        st.markdown('<div style="max-height:400px; overflow-y:auto;">', unsafe_allow_html=True)
        for i, turn in enumerate(history):
            if turn["role"] == "user":  # only show questions
                # Show user question as clickable expander
                with st.expander(f"{turn['content']}", expanded=False):
                    # Find the next assistant reply (if any)
                    if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                        st.markdown(f"🤖 {history[i + 1]['content']}")
                    else:
                        st.caption("No answer yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("No history yet.")


# Input
user_input = st.chat_input("Ask anything...")
if user_input:
    # Add user query to history immediately (green bubble)
    history.append({"role": "user", "content": user_input})
    save_chat_history(username, history)

    # Refresh chat display
    st.markdown('<div class="chat-wrapper"><div class="chat-container chat-scroll">', unsafe_allow_html=True)
    for turn in history[-50:]:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            st.markdown(f'<div class="user-msg">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{content}</div>', unsafe_allow_html=True)

    # Add a temporary "Bot is typing..." bubble
    typing_placeholder = st.empty()
    typing_placeholder.markdown('<div class="bot-msg"><em>Bot is typing...</em></div>', unsafe_allow_html=True)

    # Process with chain
    if chain_key not in st.session_state:
        typing_placeholder.empty()
        st.warning("No index found. Please upload and process PDFs first.")
    else:
        chain = st.session_state[chain_key]
        with st.spinner("Thinking..."):
            try:
                result = chain({"question": user_input})
                answer = result.get("answer") or result.get("result")

                # Remove "typing..." bubble
                typing_placeholder.empty()

                # Save bot reply
                history.append({"role": "assistant", "content": answer})
                save_chat_history(username, history)

                # Show bot reply bubble
                st.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)

                # Sources section
                sources = result.get("source_documents", [])
                if sources:
                    with st.expander("🔎 Sources"):
                        for i, doc in enumerate(sources):
                            filename = doc.metadata.get("source_file", "Unknown file")
                            excerpt = doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
                            st.markdown(f"**Source {i+1} ({filename}):** {excerpt}")

            except Exception as e:
                typing_placeholder.empty()
                st.error(f"Error: {e}")

    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("---")
#st.caption("Optimized for speed: after initial indexing, answers should return in a few seconds with Mistral.")
