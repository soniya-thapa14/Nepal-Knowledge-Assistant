# src/nepal_assistant.py
"""
Nepal Knowledge Assistant - History-aware conversational AI
Combines vector search with LLM for intelligent Q&A about Nepal
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

load_dotenv()


class NepalAssistant:
    """
    Conversational assistant for Nepal tourism and culture knowledge
    
    Features:
    - History-aware conversation
    - Follow-up question handling
    - Support for both Ollama (local) and Groq (cloud) LLMs
    """
    
    def __init__(self, db_path="data/chroma_store", llm_type="ollama",model_name = None, max_history=5):
        """
        Initialize Nepal Knowledge Assistant
        
        Args:
            db_path: Path to Chroma vector database
            llm_type: "ollama" (free, local) or "groq" (fast, cloud)
            max_history: Maximum conversation turns to remember
        """
        print("🇳🇵 Initializing Nepal Knowledge Assistant...")
        
        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        
        # Initialize LLM
        self.llm_type = llm_type
        if llm_type == "groq":
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=groq_key
            )
            print("🤖 Using Groq LLM (cloud, fast)")
        else:
            self.llm = OllamaLLM(model="gpt-oss:120b-cloud", temperature=0)
            print("🤖 Using Ollama LLM (local, free)")
        
        # Conversation memory
        self.conversation_history = []
        self.max_history = max_history
        
        print("✅ Assistant ready!\n")
    
    def _reformulate_question(self, question):
        """
        Convert follow-up questions to standalone questions
        Example: "How high is it?" → "How high is Mount Everest?"
        """
        if not self.conversation_history:
            return question
        
        # Check if question seems like a follow-up
        follow_up_indicators = [
            'it', 'that', 'there', 'them', 'this', 'those',
            'what about', 'how about', 'tell me more', 'more info'
        ]
        
        is_follow_up = any(word in question.lower() for word in follow_up_indicators)
        
        if not is_follow_up:
            return question
        
        # Use LLM to reformulate with context
        recent_history = "\n".join([
            f"Q: {q}\nA: {a[:150]}..." 
            for q, a in self.conversation_history[-2:]
        ])
        
        prompt = f"""Given this conversation:
{recent_history}

User's new question: "{question}"

If this is a follow-up question (uses "it", "that", "there", etc.), rewrite it as a standalone question. Otherwise, return unchanged.

Standalone question:"""
        
        try:
            if self.llm_type == "groq":
                reformulated = self.llm.invoke(prompt).content.strip()
            else:
                reformulated = self.llm.invoke(prompt).strip()
            
            if len(reformulated) > 5 and reformulated.endswith('?'):
                print(f"💭 Understood as: {reformulated}")
                return reformulated
        except Exception as e:
            print(f"⚠️  Could not reformulate: {e}")
        
        return question
    
    def ask(self, question, k=5):
        """
        Ask the assistant a question and append curated online resources
        based on the topic.

        Args:
            question (str): User's question
            k (int): Number of relevant documents to retrieve (internal use)

        Returns:
            str: Final answer text including relevant online links
        """
        reformulated = self._reformulate_question(question)

        docs = self.vectorstore.similarity_search(reformulated, k=k)

        if not docs:
            return "I don't have information about that in my knowledge base."

        context = "\n\n".join([
            f"{doc.page_content}"
            for doc in docs
        ])

        history_text = ""
        if self.conversation_history:
            recent = self.conversation_history[-self.max_history:]
            history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent])

        prompt = f"""You are an expert assistant on Nepal tourism, culture, and travel.

    Guidelines:
    - Answer clearly and concisely
    - Use 2-4 sentences unless more detail is needed
    - Be helpful and friendly
    - If you don't know, say so

    {f"Previous Conversation:\n{history_text}\n" if history_text else ""}
    Context:
    {context}

    Question: {question}

    Answer:"""

        try:
            if self.llm_type == "groq":
                answer = self.llm.invoke(prompt).content.strip()
            else:
                answer = self.llm.invoke(prompt).strip()
        except Exception as e:
            return f"Error generating answer: {e}"
        
        


        self.conversation_history.append((question, answer))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return answer

    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")
    
    def get_history(self):
        """Get conversation history"""
        return self.conversation_history.copy()


# Example usage
if __name__ == "__main__":
    # Initialize assistant
    assistant = NepalAssistant(llm_type="ollama", max_history=5)
    
    # Test conversation
    questions = [
        "Tell me about Mount Everest",
        "How high is it?",
        "What about trekking there?",
    ]
    
    for q in questions:
        print(f"\n{'='*70}")
        print(f"🔍 Q: {q}")
        result = assistant.ask(q)
        print(f"💬 A: {result['answer']}")
        print('='*70)