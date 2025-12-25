# src/rag_chain.py
"""
Improved RAG system for Nepal Knowledge Base
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()


class NepalRAG:
    
    def __init__(self, db_path="data/chroma_store"):
        print("🇳🇵 Loading Nepal Knowledge Base...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=groq_key
        )
        
        print("✅ Ready!\n")
    
    def ask(self, question, k=5):
        """Ask with smart prompting"""
        
        # Retrieve
        docs = self.vectorstore.similarity_search(question, k=k)
        
        if not docs:
            return {'answer': "I don't have information about that.", 'sources': []}
        
        # Build context
        context = "\n\n".join([f"[Source {i+1}]\n{doc.page_content}" 
                               for i, doc in enumerate(docs)])
        
        # Smart prompt
        prompt = f"""You are an expert on Nepal. Answer the question clearly and concisely.

uidelines:
- For "What is" questions: Define it, give 2-3 key facts
- For "Tell me about": Provide overview with most important details
- For "List/Major" questions: List main items (3-5) with brief descriptions
- Use 2-4 sentences unless the question requires more detail
- Focus on CORE FACTS from the context
- Be friendly and helpful
- **When you use information from a source, mention it naturally** (e.g., "Located in the Kathmandu Valley...")
- If the context doesn't contain the answer, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.invoke(prompt).content
        
        return {
            'answer': answer,
            'sources': docs
        }


# Test
if __name__ == "__main__":
    rag = NepalRAG()
    
    questions = [
        "What is the capital of Nepal?",
        "Tell me about Mount Everest",
        "What are the major festivals in Nepal?",
        "What are Nepal's main rivers?",
        "Tell me about Nepali culture",
        "tell me about the trekking in nepal"
    ]
    
    for q in questions:
        print(f"Q: {q}")
        result = rag.ask(q)
        print(f"A: {result['answer']}\n")
        print("-" * 70 + "\n")