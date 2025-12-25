from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data_loader import load_documents

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    

    def chunk_documents(self, documents: List[Document]) -> List[Document]:

        print("="*70)
        print("SPLITTING DOCUMENTS INTO CHUNKS")
        print("="*70)
        print(f"\nChunk size: {self.chunk_size} characters")
        print(f"Chunk overlap: {self.chunk_overlap} characters\n")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks


    