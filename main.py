# main.py
from src.data_loader import load_documents
from src.chunking import DocumentChunker
from src.embedding import EmbeddingModel
from src.vectorstore import ChromaVectorStore

def main():
    # 1. Load documents
    print("\n" + "="*70)
    print("STEP 1: LOADING DOCUMENTS")
    print("="*70)
    docs = load_documents("data")
    
    # 2. Chunk documents
    print("\n" + "="*70)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("="*70)
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(docs)
    
    # 3. Create embedding model
    print("\n" + "="*70)
    print("STEP 3: LOADING EMBEDDING MODEL")
    print("="*70)
    embedding_model = EmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 4. Build vector store
    print("\n" + "="*70)
    print("STEP 4: BUILDING VECTOR STORE")
    print("="*70)
    store = ChromaVectorStore(
        embedding_function=embedding_model.get_embedding_function(),
        persist_dir="data/chroma_store"
    )
    store.build_vectorstore(chunks)
    
    # 5. Test query
    print("\n" + "="*70)
    print("STEP 5: TESTING QUERY")
    print("="*70)
    query = "best trekking routes in Nepal"
    results = store.query(query, top_k=3)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📊 Found {len(results)} results\n")
    
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Text: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    
    print("="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()

