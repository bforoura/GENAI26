import streamlit as st
import os
from typing import Annotated, TypedDict, Union
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# --- 1. TOOL CONFIGURATION ---
# Using the parameters from your original notebook
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki_tool, arxiv_tool]

# --- 2. MODEL SETUP ---
# Local Ollama instance with tool binding
llm = ChatOllama(
    model="qwen2.5-coder:3b", 
    temperature=0
).bind_tools(tools)

# --- 3. GRAPH STATE DEFINITION ---
class AgentState(TypedDict):
    # add_messages is a specialized reducer that appends new messages to history
    messages: Annotated[list[BaseMessage], add_messages]

# --- 4. NODE FUNCTIONS ---
def call_model(state: AgentState):
    """
    The 'Thought' Node.
    The model examines the message history and decides whether 
    to call a tool or provide a final answer.
    """
    messages = state['messages']
    
    # Optional: Inject system instructions if this is the start of the thread
    if len(messages) <= 1:
        system_prompt = SystemMessage(
            content="You are a helpful research assistant. Use Wikipedia for general facts and Arxiv for scientific papers."
        )
        messages = [system_prompt] + messages
        
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- 5. EDGE LOGIC (ROUTING) ---
def should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    last_message = state['messages'][-1]
    
    # If the LLM didn't request a tool, we finish the loop
    if not last_message.tool_calls:
        return END
    
    # Otherwise, we route to the 'tools' node
    return "tools"

# --- 6. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Define the two nodes in our cycle
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Define the logic flow
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # Loops back after tool execution

# Compile into a runnable application
app = workflow.compile()

# --- 7. STREAMLIT INTERFACE ---
st.set_page_config(page_title="LangGraph Research Agent", layout="wide")
st.title("Modern Research Agent: LangGraph + Ollama")

# Initialize session state for persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif msg.content: # Only display visible text, skip raw tool call metadata
        with st.chat_message("assistant"):
            st.write(msg.content)

# Handle User Input
if user_query := st.chat_input("What would you like to research?"):
    # Add user message to state
    new_user_msg = HumanMessage(content=user_query)
    st.session_state.messages.append(new_user_msg)
    
    with st.chat_message("user"):
        st.write(user_query)

    # Run the Graph
    with st.chat_message("assistant"):
        # We start the graph with the current message history
        inputs = {"messages": st.session_state.messages}
        
        # Stream the graph updates
        # 'values' mode provides the full state at each step
        for output in app.stream(inputs, stream_mode="values"):
            latest_msg = output["messages"][-1]
            
        # Final answer display
        st.write(latest_msg.content)
        st.session_state.messages.append(latest_msg)