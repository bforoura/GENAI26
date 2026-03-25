import streamlit as st
import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE SETUP ---
st.set_page_config(page_title="Local Ollama Chat", layout="wide")
st.title("🦙 Local Ollama Chatbot")

# --- SIDEBAR: MODEL SELECTION ---
with st.sidebar:
    st.header("Settings")
    
    try:
        # UPDATED: Correct way to access model names in newer ollama-python versions
        response = ollama.list()
        
        # Accessing the 'models' attribute and then the 'model' property of each object
        model_names = [m.model for m in response.models]
        
        # Filter out embedding models
        chat_models = [n for n in model_names if "embed" not in n.lower()]
        
        selected_model = st.selectbox(
            "Select your LLM:",
            options=chat_models,
            index=0 if chat_models else None
        )
    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")
        selected_model = None

        
# --- CHAT LOGIC ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Say something..."):
    if not selected_model:
        st.warning("Please select a model from the sidebar first.")
    else:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Initialize LangChain ChatOllama
            llm = ChatOllama(model=selected_model, streaming=True)
            
            # Convert session history to LangChain format
            langchain_messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages
            ]

            # Stream the response
            try:
                for chunk in llm.stream(langchain_messages):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Add assistant response to state
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")