📚 Q&A Documentation Chatbot

A conversational chatbot designed to answer documentation-based questions using Ollama (Mistral), LangChain, Streamlit, and ChromaDB.

🚀 Features

🤖 LLM-powered Responses using Ollama (Mistral model)

🔗 LangChain Integration for building retrieval-augmented generation (RAG) pipelines

📂 ChromaDB as vector database for storing embeddings

🌐 Interactive UI with Streamlit

🔍 Semantic Search over documentation for accurate answers , Multi-document support

🛠️ Tech Stack

Frontend/UI: Streamlit

Backend/Logic: Python, LangChain

LLM Engine: Ollama (Mistral)

Database: ChromaDB (vector storage)

📂 Project Structure
chatbot-project/
│── app.py              # Streamlit app entry point
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/chatbot-project.git
cd chatbot-project


Create a virtual environment & install dependencies

pip install -r requirements.txt


Run Ollama with Mistral model

ollama run mistral


Start the Streamlit app

streamlit run app.py

💡 Usage

Upload or connect documentation files into the docs/ folder

The chatbot indexes them into ChromaDB

Ask any question in the Streamlit UI and get relevant answers

🔮 Future Enhancements

Enable fine-tuning with domain-specific data

🤝 Contribution

Feel free to fork, open issues, or submit PRs to improve this project.
