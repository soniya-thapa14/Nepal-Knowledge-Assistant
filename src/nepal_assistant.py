import os
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

load_dotenv()


class NepalAssistant:
    """Nepal Knowledge Assistant"""
    
    def __init__(self, db_path="data/chroma_store", llm_type="ollama", 
                 model_name=None, max_history=5, enable_web_search=True):
        print("🇳🇵 Initializing Nepal Knowledge Assistant...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        
        # Initialize both LLMs
        self.llm_type = llm_type
        groq_key = os.environ.get("GROQ_API_KEY")
         
        # Groq LLM
        self.groq_llm = None
        if groq_key:
            try:
                self.groq_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    groq_api_key=groq_key
                )
                print("🤖 Groq available")
            except:
                pass
        
        # Ollama LLM
        self.ollama_llm = None
        try:
            ollama_model = model_name or "gpt-oss:120b-cloud"
            self.ollama_llm = OllamaLLM(model=ollama_model, temperature=0)
            print(f"🤖 Ollama available: {ollama_model}")
        except:
            pass
        
        # Set primary LLM
        if llm_type == "groq" and self.groq_llm:
            self.llm = self.groq_llm
        elif self.ollama_llm:
            self.llm = self.ollama_llm
        else:
            self.llm = self.groq_llm
        
        if not self.llm:
            raise ValueError("No LLM available!")
        
        # Web search setup
        self.enable_web_search = enable_web_search
        self.serper_api_key = os.environ.get("SERPER_API_KEY")
        
        if enable_web_search:
            if self.serper_api_key:
                print("🌐 Web search enabled (Serper - Google Results)")
            else:
                print("⚠️ SERPER_API_KEY not found, web search disabled")
                self.enable_web_search = False
        
        self.conversation_history = []
        self.max_history = max_history
        print("✅ Ready!\n")

    def _is_follow_up(self, question):
        if not self.conversation_history:
            return False

        recent_history = "\n".join([
            f"Q: {q}\nA: {a[:150]}" for q, a in self.conversation_history[-2:]
        ])

        prompt = f"""Recent conversation:
    {recent_history}

    New question: "{question}"

    Does this question depend on the previous conversation to make sense?

    Answer YES if:
    - Uses pronouns like him, her, they, it, this, that
    - Refers to something without naming it
    - Would be unclear without conversation history

    Answer NO if:
    - It is a complete standalone question

    Answer ONLY: YES or NO"""

        try:
            if self.groq_llm:
                response = self.groq_llm.invoke(prompt).content.strip().upper()
            else:
                response = self.ollama_llm.invoke(prompt).strip().upper()
            return "YES" in response
        except:
            return False
        
    def _reformulate_question(self, question):
        if not self._is_follow_up(question):
            return question

        recent_history = "\n".join([
            f"Q: {q}\nA: {a[:150]}..." for q, a in self.conversation_history[-2:]
        ])

        prompt = f"""Rewrite the user's question as a standalone question
                if it depends on previous context. If the user says "them", "they", or similar 
                and the conversation mentions multiple people, include all of them.
                If someone held multiple terms, reference their most recent term.
    Conversation:
    {recent_history}

    User's new question: "{question}"

    Write ONLY the standalone question, nothing else:"""

        try:
            if isinstance(self.llm, ChatGroq):
                reformulated = self.llm.invoke(prompt).content.strip()
            else:
                reformulated = self.llm.invoke(prompt).strip()

            if len(reformulated) > 5:
                print(f"💭 Reformulated: {reformulated}")
                return reformulated
        except Exception as e:
            print(f"⚠️ Could not reformulate: {e}")

        return question

    
    def _search_serper(self, query):
        """Search using Serper.dev (Google Results)"""
        try:
            url = "https://google.serper.dev/search"
            
            payload = {
                "q": f"{query} Nepal 2026",  # Add year for recent results
                "gl": "np",  # Nepal
                "hl": "en",  # English
                "num": 5     # Top 5 results
            }
            
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            results = []
            images = []
            
            # Check for answer box (featured snippet)
            if "answerBox" in data:
                answer_box = data["answerBox"]
                if "answer" in answer_box:
                    results.append(f"**Featured Answer:**\n{answer_box['answer']}")
                elif "snippet" in answer_box:
                    results.append(f"**Featured Snippet:**\n{answer_box['snippet']}")
            
            # Add organic search results
            if "organic" in data:
                for item in data["organic"][:5]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    date = item.get("date", "")

                    
                    result_text = f"**{title}**"
                    if date:
                        result_text += f" ({date})"
                    result_text += f"\n{snippet}\nSource: {link}"
                    results.append(result_text)

            if self.needs_image(query):
                image_url = "https://google.serper.dev/images"
                img_response = requests.post(image_url, json = payload, headers = headers)
                img_data = img_response.json()
                print(f"DEBUG: img_data keys: {img_data.keys()}")  # ADD THIS LINE
                print(f"DEBUG: full response: {img_data}")  # ADD THIS TOO
    

                if "images" in img_data:
                    for img in img_data["images"][:2]:
                        images.append(img.get("imageUrl"))

                    print(f"found {len(images)} images")
            return ("\n\n".join(results) if results else None, images)
            
        except Exception as e:
            print(f"⚠️ Serper search failed: {e}")
            return (None, [])
    
    def needs_image(self,question):
        image_words =["image", "photo", "picture","pic", "show me"]
        q= question.lower()
        return any(word in q for word in image_words)
        
    def _search_web(self, question):
        """Perform web search"""
        print("🔍 Searching web for current information...")
        return self._search_serper(question)

    def ask(self, question, k=5):

        greetings = ["how are you", "hello", "hi", "hey", "namaste", "good morning", "good evening"]
        q = question.lower().strip()
        if q in greetings:
            return {
                'answer': "Namaste! 🙏 I'm here to help you learn about Nepal. What would you like to know? 🏔️",
                'sources': [],
                'used_web_search': False,
                'llm_used': 'none',
                'images': []
            }

        """Ask with smart LLM selection: Ollama for web search, Groq for regular"""
        reformulated = self._reformulate_question(question)
        needs_image = self.needs_image(reformulated)

        is_time_sensitive = self.time_sensitive_questions(reformulated)

        docs = self.vectorstore.similarity_search(reformulated, k =k)
        
        if is_time_sensitive:
            print("Time sensitive query")
            use_web = True
        elif needs_image:
            print("Image request")
            use_web = True
        elif not docs or sum(len(doc.page_content) for doc in docs) <100:
            print("Weak local answer")
            use_web = True
        else:
            print("strong local answer")
            use_web = False
            
        # Switch LLM based on web search need
        if use_web:
            if self.groq_llm:
                active_llm = self.groq_llm
                llm_type = "groq"
                print("🎯 Using Groq for web search (safe for hosting)")
            elif self.ollama_llm:
                active_llm = self.ollama_llm
                llm_type = "ollama"
                print("🎯 Using Ollama for web search")
        else:
            if self.groq_llm:
                active_llm = self.groq_llm
                llm_type = "groq"
                print("🎯 Using Groq for local question")
            elif self.ollama_llm:
                active_llm = self.ollama_llm
                llm_type = "ollama"
        # BUid contexts
        local_context = "\n\n".join([
            f"[Local Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        ]) if docs else "No relevant local information found."
        
        web_context = ""
        images = []
        if use_web:
            web_results, images = self._search_web(reformulated)
            if web_results:
                web_context = f"\n\n[Web Search Results - Google (via Serper)]\n{web_results}"
        
        full_context = local_context + web_context

        history_text = ""
        if self.conversation_history:
            recent = self.conversation_history[-self.max_history:]
            history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent])

        prompt = f"""You are the Nepal Knowledge Assistant. You are polite, knowledgeable, and direct.

{f"Previous Conversation:\n{history_text}\n" if history_text else ""}

Information Sources:
{full_context}

Question: {reformulated}

CRITICAL INSTRUCTIONS:
1. Use Web Search for current events (2025-2026) and Local Sources for culture/geography.
2. Do not say "Searching...", "I found...", or "According to the sources".
3. Do not list URLs or sources in your text.
4. If images are found, provide exactly a 1-2 sentence description of the subject.
5. 2-3 sentences maximum. Give the direct answer first, then one supporting fact only.
6. Do NOT mention web search or sources
7. If the question is completely unrelated to Nepal, politely say "I'm only able to answer questions about Nepal. What would you like to know about Nepal?
8. Never make subjective comparisons or say one place is "better" than another. Stay neutral and factual. 
Answer:"""

        try:
            if llm_type == "groq":
                answer = active_llm.invoke(prompt).content.strip()
            else:
                answer = active_llm.invoke(prompt).strip()
        except Exception as e:
            return {
                'answer': f"Error: {e}",
                'sources': docs,
                'used_web_search': use_web,
                'llm_used': llm_type,
                'images' : []
            }

        self.conversation_history.append((question, answer))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return {
            'answer': answer,
            'sources': docs,
            'used_web_search': use_web,
            'llm_used': llm_type,
            'images': images
        }
    def time_sensitive_questions(self,question):
        return self.ask_llm_is_time_sensitive(question)
    
    def ask_llm_is_time_sensitive(self,question):
        context = ""
        if self.conversation_history:
            recent = self.conversation_history[-2:]
            context = "Recent conversation:\n" + "\n".join(
                [f"Q: {q}\nA: {a[:150]}" for q, a in recent]
            )
        prompt = f"""{context}
        Question": '{question}'

    Does this question require searching the web to answer accurately?

Answer YES if:
- Asks about current/recent positions, roles, events
- Asks about who came before/after someone in a role
- Asks about results, scores, news, announcements
- The answer could have changed in the last year
- Needs specific real-world data that a static knowledge base might not have

Answer NO if:
- Asks about historical facts, geography, culture, definitions
- The answer is stable and unlikely to change

Answer ONLY: YES or NO"""
        
        try:
            if self.groq_llm:
                response = self.groq_llm.invoke(prompt).content.strip().upper()
            else:
                response = self.ollama_llm.invoke(prompt).strip().upper()
            print(f"DEBUG LLM time-sensitive check: '{response}'")  
            return "YES" in response
        except Exception as e:
            print(f" Time-sensitivity check failed: {e}, defaulting to YES")
            return True


    def clear_history(self):
        self.conversation_history = []
        print("🗑️ History cleared")

    def get_history(self):
        return self.conversation_history.copy()


if __name__ == "__main__":
    assistant = NepalAssistant(enable_web_search=True)
    
    question = [
        "What is the capital of Japan?",
        "Tell me about Elon Musk",
        "What is Bitcoin?"
            ]

    for q in question:
        print(f"\n{'='*70}")
        result = assistant.ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Used web search: {result['used_web_search']}")
        print(f"LLM used: {result['llm_used']}")
        print('='*70)