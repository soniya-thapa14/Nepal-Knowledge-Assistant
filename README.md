🇳🇵 Nepal Knowledge Assistant
A RAG chatbot that answers questions about Nepal using a local vector database and real-time web search.

Features

Features

- Hybrid RAG (ChromaDB + real-time Google search via Serper)
- Fast inference using Groq (LLaMA 3.3 70B)
- Conversation memory (last 5 turns)
- Follow-up question detection & reformulation
- Image retrieval support
- Domain-restricted to Nepal queries

Stack

- LangChain — RAG orchestration
- ChromaDB — persistent vector store
- HuggingFace all-MiniLM-L6-v2 — embeddings
- Groq / llama-3.3-70b-versatile — LLM
- Serper.dev — web and image search

Setup

Clone the repo

git clone https://github.com/soniya-thapa14/Nepal-Knowledge-Assistant.git


Install dependencies

pip install -r requirements.txt

Create a .env file

GROQ_API_KEY=your_groq_key
SERPER_API_KEY=your_serper_key

streamlit run app.py
