import streamlit as st
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import requests



######################################################################################
# Define the Weather Tool
######################################################################################
@tool
def get_weather(location: str) -> str:
    """
    Fetches the current weather for a given city name. 
    It first geocodes the city to coordinates and then retrieves the weather data.
    """
    try:
        # Step 1: Geocoding - Convert city name to lat/long
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=10).json()
        
        if not geo_res.get("results"):
            return f"Could not find coordinates for {location}. Please check the spelling."
        
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        name = geo_res["results"][0]["name"]
        
        # Step 2: Get weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url, timeout=10).json()
        
        current = weather_res.get("current_weather")
        if not current:
            return "Weather data is currently unavailable."
            
        temp = current["temperature"]
        wind = current["windspeed"]
        
        return f"The current weather in {name} is {temp} degrees Celsius with a wind speed of {wind} km/h."
    
    except Exception as e:
        return f"Error retrieving weather: {str(e)}"



######################################################################################
# Setup the Streamlit UI
######################################################################################
st.set_page_config(page_title="Weather ReAct Agent", layout="wide")
st.title("Local Weather Agent")
st.caption("Using Ollama and Open-Meteo API")

with st.sidebar:
    model_choice = st.selectbox(
        "Select Model", 
        ["qwen2.5-coder:7b", "qwen2.5-coder:3b", "llama3.2:3b", "deepseek-r1:1.5b"]
    )
    st.info("Ask about the weather in any city. No API key required.")




######################################################################################
# Initialize the Agent
######################################################################################
llm = ChatOllama(model=model_choice, temperature=0)
tools = [get_weather]



######################################################################################
# Standard ReAct Prompt
######################################################################################
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



######################################################################################
# Create the agent logic
######################################################################################
agent = create_react_agent(llm, tools, prompt)


######################################################################################
# Create the executor
######################################################################################
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)




######################################################################################
# Chat Interface
######################################################################################
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is the weather in Tokyo?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting weather stations..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                final_answer = response["output"]
            except Exception as e:
                final_answer = f"An error occurred during agent execution: {str(e)}"
            
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})