import streamlit as st
from langchain_ollama import ChatOllama
# Corrected Imports for LangChain v1.0+ compatibility
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.prompts import PromptTemplate


############################################################################################
# Initialize Research Tools
# Using wrappers to limit the data sent to the local LLM
############################################################################################
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Tool list restricted to Wikipedia and Arxiv
tools = [wiki_tool, arxiv_tool]




############################################################################################
# Agent Configuration
# Ensure Ollama is running locally with the qwen2.5-coder:3b model
############################################################################################
llm = ChatOllama(model="qwen2.5-coder:3b", temperature=0)



############################################################################################
# ReAct Prompt Template
############################################################################################
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Build the ReAct Agent using the modern constructor
agent = create_react_agent(llm, tools, prompt)



############################################################################################
# Create the executor using the classic compatibility layer
############################################################################################
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)



############################################################################################
# Streamlit User Interface
############################################################################################
st.set_page_config(page_title="Local Research Assistant", layout="wide")
st.title("Local AI Research Agent")
st.markdown("This agent uses Wikipedia for general facts and Arxiv for scientific papers.")



############################################################################################
# Initialize session state for chat history
############################################################################################
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User interaction logic
if user_input := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching sources..."):
            try:
                # Invoke the agent executor
                response = agent_executor.invoke({"input": user_input})
                final_answer = response["output"]
            except Exception as e:
                final_answer = f"Error during agent execution: {str(e)}"
            
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})