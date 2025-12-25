# app.py
import os
from dotenv import load_dotenv
from src.nepal_assistant import NepalAssistant

load_dotenv()

def main():
    print("\n🇳🇵 NEPAL KNOWLEDGE ASSISTANT 🇳🇵\n")
    
    # Auto-detect LLM
    if os.getenv("GROQ_API_KEY"):
        print("✅ Using Groq (Fast)")
        llm_type = "groq"
        model_name = None
    else:
        print("✅ Using Ollama (Local)")
        llm_type = "ollama"
        model_name = "gpt-oss:120b-cloud "  # ✅ Correct model name
    
    print("\n⏳ Initializing...\n")
    
    assistant = NepalAssistant(
        llm_type=llm_type,
        model_name=model_name,  # ✅ Pass the model name
        max_history=5
    )
    
    print("="*70)
    print("✅ Ready! Ask me anything about Nepal")
    print("="*70 + "\n")
    
    while True:
        question = input("🔍 Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Namaste!\n")
            break
        
        if question.lower() == 'clear':
            assistant.clear_history()
            continue
        
        print("\n🤔 Thinking...\n")
        result = assistant.ask(question, k=3)
        
        print("💬 Answer:")
        print("-" * 70)
        print(result['answer'])
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()