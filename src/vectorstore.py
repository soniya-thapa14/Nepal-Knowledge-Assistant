# src/vectorstore.py
from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.data_loader import load_documents
from src.chunking import DocumentChunker
from src.embedding import EmbeddingModel

class ChromaVectorStore:
    def __init__(self, embedding_function, persist_dir: str = "data/chroma_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self.embeddings = embedding_function
        self.vectorstore = None
        print(f"✅ Vector store initialized at {self.persist_dir}")

    def build_vectorstore(self, chunks: List[Document]):
        """Create Chroma vector store."""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        self.vectorstore.persist()
        print(f"✅ Vector store created with {len(chunks)} vectors")

    def load_vectorstore(self):
        """Load existing vector store from disk."""
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )
        print(f"✅ Loaded vector store from {self.persist_dir}")

    def query(self, query_text: str, top_k: int = 5):
        """Perform semantic search."""
        if not self.vectorstore:
            raise ValueError("Vector store not loaded!")
        return self.vectorstore.similarity_search(query_text, k=top_k)
    
