# src/embedding.py

from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Loaded embedding model: {model_name}")
    
    def get_embedding_function(self):
        return self.embeddings
    
