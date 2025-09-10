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
# User-configurable
# ------------------------
OLLAMA_BASE_URL = "http://localhost:11434"  # change if your Ollama server uses a different host/port
EMBED_MODEL = "nomic-embed-text"  # recommended embedding model; replace if you have another
LLM_MODEL = "phi3"  # change to the LLM you pulled and want to use
PERSIST_DIR = "./chroma_db"  # where Chroma will persist vectors (optional)

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="PDF QA Chatbot", page_icon="🤖")
st.title("📚 PDF QA Chatbot 🤖")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# helper to show debugging info
def show_debug(msg: str):
    st.info(msg)

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load PDF (returns list[Document])
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        if not docs:
            st.error("No pages were extracted from the PDF. If the PDF is scanned/images-only, try a loader that does OCR (UnstructuredPDFLoader or OCR pipeline).")
            st.stop()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        # ------------------
        # Create Ollama embeddings (validate)
        # ------------------
        with st.spinner("Creating embeddings via Ollama..."):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

            # Quick validation: embed a small query to make sure Ollama returns a non-empty vector
            try:
                sample = embeddings.embed_query("hello world")
            except Exception as e:
                st.error(f"Failed to call Ollama embeddings: {e}")
                st.stop()

            if not sample or len(sample) == 0:
                st.error(
                    "Received empty embedding vector from Ollama.\n" +
                    "→ Possible causes: wrong embedding model name, Ollama server not running, or the model does not provide embeddings.\n" +
                    "Try: `ollama list` and `ollama pull nomic-embed-text` (or whichever embedding model you want) and ensure the Ollama daemon is running."
                )
                st.stop()

        # ------------------
        # Create / persist Chroma DB
        # ------------------
        with st.spinner("Indexing documents into Chroma (this may take a while)..."):
            db = Chroma.from_documents(documents=documents, embedding=embeddings)
            # persist to disk so you don't need to re-create embeddings every reload
            try:
                db.persist()
            except Exception:
                # persist may not be available in some setups; ignore if not
                pass

        retriever = db.as_retriever()

        # Initialize Ollama LLM (use same base_url)
        llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

        # Prompt template for RAG
        prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing an answer. 
<context>
{context}
</context>
Question: {input}
""")

        # Stuff documents chain (combines docs into a prompt and calls the LLM)
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Retrieval chain: retriever -> document_chain
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
                    st.error(f"Error while running retrieval chain: {e}")
                    st.stop()

                # create_retrieval_chain returns at least 'answer' and 'context'
                answer = result.get("answer")
                context_docs = result.get("context")

                if answer is None:
                    st.error("Chain finished but no 'answer' key was returned. Inspect `result` for details.")
                    st.write(result)
                else:
                    st.markdown(f"**Question:** {query}")
                    st.markdown(f"**Answer:** {answer}")

                    # show short snippets of the retrieved context so you can validate retrieval
                    if context_docs:
                        st.subheader("Retrieved context (first 3 docs):")
                        for i, d in enumerate(context_docs[:3]):
                            # each item is a Document-like object; handle both string or Document
                            try:
                                content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                            except Exception:
                                content = str(d)
                            st.write(f"--- doc {i+1} ---")
                            st.write(content[:1000])

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        raise

else:
    st.info("👆 Please upload a PDF file to start chatting.")
