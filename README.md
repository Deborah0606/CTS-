# QnA-ChatBot

An AI-powered PDF assistant using **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers from your documents.



## PDF QA Chatbot (RAG-based LLM)

A **RAG-powered chatbot** that allows you to upload any PDF and get **accurate, context-aware answers** to your questions. Built using **LangChain**, **Ollama Phi-3**, **Chroma**, and **Streamlit**.



## Features

- 📄 **PDF Upload & Parsing**: Upload any PDF, automatically split into chunks for semantic retrieval.  
- 🧠 **Vector Embeddings**: Uses `OllamaEmbeddings` to convert document chunks into vectors for retrieval.  
- 🔍 **RAG-based Question Answering**: Answers questions based solely on the content of the uploaded PDF.  
- 🖥 **Interactive UI**: Streamlit interface for easy question input and answer display.  



## Tech Stack

- **Python**  
- **Streamlit** for UI  
- **LangChain** for RAG pipelines  
- **Ollama Phi-3** for embeddings & LLM  
- **Chroma** as vector store  



## Usage

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```



### 2. Install dependencies:

```bash
pip install -r requirements.txt
```


### 3. Install Ollama and download Phi-3 model:
Note: The app uses a local Ollama model, so ensure Phi-3 is installed on the machine where you run the app.

```bash
ollama pull phi3
```

### 4. Run the app:

```bash
streamlit run app.py
```

### 5. Upload a PDF and start asking questions!
