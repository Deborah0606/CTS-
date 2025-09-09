import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="PDF QA Chatbot", page_icon="🤖")
st.title("📚 PDF QA Chatbot 🤖")

# ========================
# FILE UPLOAD
# ========================
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # Create vector store with Ollama embeddings
    embeddings = OllamaEmbeddings(model="phi3")
    db = Chroma.from_documents(documents, embeddings)

    # Initialize Ollama LLM
    llm = OllamaLLM(model="phi3")

    # Prompt template for RAG
    prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing an answer. 
<context>
{context}
</context>
Question: {input}
""")

    # Stuff documents chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever
    retriever = db.as_retriever()

    # Retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ========================
    # CHAT UI
    # ========================
    query = st.text_input("Ask something about your PDF...")

    if query:
        result = retrieval_chain.invoke({"input": query})
        answer = result["answer"]

        st.markdown(f"**Question:** {query}")
        st.markdown(f"**Answer:** {answer}")

else:
    st.info("👆 Please upload a PDF file to start chatting.")