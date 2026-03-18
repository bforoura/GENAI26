import streamlit as st
from langchain_ollama import ChatOllama

# Application Title
st.set_page_config(page_title="Local LLM Playground", layout="centered")
st.title("Local LLM Chat")

# Model options from your local 'ollama list'
model_options = [
    "deepseek-r1:1.5b",
    "smollm2:1.7b",
    "llama3.2:1b",
    "llama3.2:3b",
    "mistral-nemo:latest",
    "phi3.5:latest",
    "llama3.1:latest",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "gemma2:2b",
    "mapler/gpt2:latest"
]

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Choose your model:", model_options)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    if st.button("Clear Chat"):
        st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is on your mind?"):
    # Add user message to history
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        try:
            chat = ChatOllama(
                model=selected_model,
                temperature=temperature,
                base_url="http://127.0.0.1:11434"
            )
            
            # Prepare messages for LangChain
            formatted_messages = [
                ("system", "You are a helpful assistant.")
            ]
            for m in st.session_state.messages:
                formatted_messages.append((m["role"], m["content"]))

            # Stream the response for a better UI experience
            response_container = st.empty()
            full_response = ""
            
            for chunk in chat.stream(formatted_messages):
                full_response += chunk.content
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")