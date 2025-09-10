# app.py - Finalized PDF QA Chatbot (Streamlit + Ollama + Chroma)
import os
import traceback
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ------------------------
# Default user-configurable (can be overridden via environment / Streamlit Secrets)
# ------------------------
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "phi3"
DEFAULT_PERSIST_DIR = "./chroma_db"  # optional persistence dir

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="PDF QA Chatbot", page_icon="🤖")
st.title("📚 PDF QA Chatbot 🤖")

# Read config from env (Streamlit Secrets -> os.environ)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")  # set this in Streamlit Secrets when deploying
EMBED_MODEL = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
PERSIST_DIR = os.environ.get("PERSIST_DIR", DEFAULT_PERSIST_DIR)

# If base URL is not set, warn and default to localhost (useful for local testing)
if not OLLAMA_BASE_URL:
    st.warning(
        "OLLAMA_BASE_URL not found in environment. Defaulting to http://localhost:11434.\n"
        "If you're running on Streamlit Cloud, set OLLAMA_BASE_URL in Secrets to your ngrok/VPS URL."
    )
    OLLAMA_BASE_URL = "http://localhost:11434"

# Show small debug info in sidebar
st.sidebar.header("Debug / Config")
st.sidebar.write("Ollama base URL:")
st.sidebar.code(OLLAMA_BASE_URL)
st.sidebar.write("Embedding model:")
st.sidebar.code(EMBED_MODEL)
st.sidebar.write("LLM model:")
st.sidebar.code(LLM_MODEL)
st.sidebar.write("Chroma persist dir:")
st.sidebar.code(PERSIST_DIR)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def show_exception(e: Exception):
    st.error(f"Unexpected error: {e}")
    with st.expander("Show traceback"):
        st.text(traceback.format_exc())

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        if not docs:
            st.error(
                "No pages were extracted from the PDF. "
                "If the PDF is scanned/images-only, try a loader that does OCR."
            )
            st.stop()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        # ------------------
        # Create Ollama embeddings (validate)
        # ------------------
        with st.spinner("Creating embeddings via Ollama..."):
            try:
                embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
            except Exception as e:
                st.error("Failed to initialize OllamaEmbeddings. Check OLLAMA_BASE_URL and model name.")
                show_exception(e)
                st.stop()

            # Quick validation: embed a sample to ensure connectivity & non-empty vector
            try:
                sample = embeddings.embed_query("hello world")
            except Exception as e:
                st.error(
                    "Failed to call Ollama embeddings: Check that Ollama is running, reachable, "
                    "and the embedding model is available on the server."
                )
                show_exception(e)
                st.stop()

            if not sample or len(sample) == 0:
                st.error(
                    "Received empty embedding vector from Ollama.\n"
                    "Possible causes: wrong embedding model name, Ollama server not running, or the model does not provide embeddings.\n"
                    "Try: `ollama list` and `ollama pull nomic-embed-text` on the machine running Ollama."
                )
                st.stop()

        # ------------------
        # Create / persist Chroma DB (try persistent first, fallback to in-memory)
        # ------------------
        db = None
        with st.spinner("Indexing documents into Chroma (this may take a while)..."):
            try:
                # Try persistent mode (works well locally)
                db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=PERSIST_DIR)
                # attempt to persist (if supported)
                try:
                    db.persist()
                except Exception:
                    # not fatal; some installs don't expose persist
                    pass
            except Exception as e_persist:
                # fallback to in-memory Chroma (works in environments where chromadb cannot run)
                st.warning("Persistent Chroma failed — falling back to in-memory Chroma. See details in the traceback.")
                with st.expander("Chroma persistent error"):
                    st.text(str(e_persist))
                try:
                    db = Chroma.from_documents(documents=documents, embedding=embeddings)
                except Exception as e_mem:
                    st.error("Failed to create in-memory Chroma. Please check chromadb installation.")
                    show_exception(e_mem)
                    st.stop()

        # Retriever
        retriever = db.as_retriever()

        # Initialize Ollama LLM
        try:
            llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        except Exception as e:
            st.error("Failed to initialize Ollama LLM. Check OLLAMA_BASE_URL and the LLM model name.")
            show_exception(e)
            st.stop()

        # Prompt template
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context. 
Think step by step before providing an answer. 
<context>
{context}
</context>
Question: {input}
"""
        )

        # Chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # ========================
        # CHAT UI
        # ========================
        query = st.text_input("Ask something about your PDF...")

        if query:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = retrieval_chain.invoke({"input": query})
                except Exception as e:
                    st.error("Error while running retrieval chain.")
                    show_exception(e)
                    st.stop()

                # Try a few common keys for returned text
                answer = None
                for key in ("answer", "output_text", "text", "response"):
                    if isinstance(result, dict) and key in result:
                        answer = result.get(key)
                        break

                # If the chain returns result['result'] or nested structure, fallback to printing result
                if answer is None:
                    # as a helpful fallback, attempt to join any string values
                    if isinstance(result, dict):
                        joined = " ".join(str(v) for v in result.values() if isinstance(v, (str, int, float)))
                        answer = joined or None

                if not answer:
                    st.warning("No plain 'answer' text was returned by the chain. Inspecting full result below.")
                    st.write(result)
                else:
                    st.markdown(f"**Question:** {query}")
                    st.markdown(f"**Answer:** {answer}")

                # Display retrieved docs (validate retrieval)
                try:
                    # get the raw docs from retriever to show snippets
                    docs_for_display = retriever.get_relevant_documents(query)
                except Exception:
                    docs_for_display = None

                if docs_for_display:
                    st.subheader("Retrieved context (first 3 docs):")
                    for i, d in enumerate(docs_for_display[:3]):
                        try:
                            content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                        except Exception:
                            content = str(d)
                        st.write(f"--- doc {i+1} ---")
                        st.write(content[:1000])

    except Exception as e:
        show_exception(e)

else:
    st.info("👆 Please upload a PDF file to start chatting.")

