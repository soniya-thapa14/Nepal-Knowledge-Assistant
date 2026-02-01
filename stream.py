# streamlit_app.py
"""
Streamlit Chat Interface for Nepal Knowledge Assistant
"""
import streamlit as st
import os
from dotenv import load_dotenv
from src.nepal_assistant import NepalAssistant

load_dotenv()

# Page config
st.set_page_config(
    page_title="Nepal Knowledge Assistant",
    page_icon="🇳🇵",
    layout="wide"
)

# Title
st.title("🇳🇵 Nepal Knowledge Assistant")
st.markdown("Namaste! 🙏 Wanna know about Nepal? I'm here to help! 🏔️")
# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # LLM Selection
    if os.getenv("GROQ_API_KEY"):
        llm_type = "groq"

    else:
        llm_type = "ollama"
        st.warning("💡 Add GROQ_API_KEY to .env for faster responses")
    
    
    max_history = 10
    
    if st.button("🗑️ Clear History"):
        if 'assistant' in st.session_state:
            st.session_state.assistant.clear_history()
        st.session_state.messages = []
        st.success("Conversation cleared!")
    
    st.markdown("---")
    st.markdown("### 📚 Example Questions")
    examples = [
        "What is Mount Everest?",
        "Tell me about trekking in Nepal",
        "What are the major festivals?",
        "Best time to visit Nepal?",
        "What permits do I need?"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.example_query = ex

# Initialize assistant (only once)
def get_assistant(llm_type,max_history):
    try:
        return NepalAssistant(llm_type=llm_type,
                              model_name= "gpt-oss:120b-cloud" if llm_type == "ollama" else None,
                              max_history= max_history)
    except Exception as e:
        st.error(f"❌ Error initializing assistant: {e}")
        if llm_type == "ollama":
            st.error("💡 Install Ollama: `winget install Ollama.Ollama`")
        return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'assistant' not in st.session_state:
    with st.spinner("🔄 Loading Nepal Knowledge Base..."):
        st.session_state.assistant = get_assistant(llm_type, max_history)

assistant = st.session_state.assistant

if assistant is None:
    st.error("❌ Assistant could not be initialized. Check settings.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If previous messages had URLs, you can skip or handle separately

# Handle example query button
query = None
# Check if there's an example query to process
if 'example_query' in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query
    
# Always show the chat input for continuous conversation
user_input = st.chat_input("Ask me about Nepal...")

# If there's new user input, use that as the query
if user_input:
    query = user_input

# Process user input
if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                if 'image' in query.lower() or 'photo' in query.lower():
                    original_llm = assistant.llm
                    if assistant.ollama_llm:
                        assistant.llm = assistant.ollama_llm
                                # Ask assistant
                result = assistant.ask(query, k=3)
                print(f"STREAMLIT DEBUG - Images: {result.get('images')}")

                # Display assistant answer
                st.markdown(result['answer'])

                if result.get('images'):
                    st.markdown("🖼️ Images:")
                    for img_url in result['images']:
                        try:
                            st.image(img_url, width = 'stretch')
                        except:
                            st.markdown(f"[View image]({img_url})")
                

                # Append message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer']
                })

                # ---- Handle curated online resources ----
                from src.topic_router import detect_topic
                from src.online_sources import ONLINE_SOURCES

                topic = detect_topic(query)
                if topic:
                    links = ONLINE_SOURCES.get(topic, [])
                    if links:
                        with st.expander("📚 Online Sources"):
                            
                            for link in links:
                                st.markdown(f"- **[{link['name']}]({link['url']})**: {link['desc']}")


            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.markdown(
    f"LLM: {llm_type.upper()} | "
    f"Messages: {len(st.session_state.messages)}"
)