import os
import streamlit as st
import asyncio
from agents import Runner, Agent, function_tool
from tavily import TavilyClient

# run this script with `streamlit run simple_chat.py`

# Get API keys from Streamlit secrets (for cloud) or environment variables (for local)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("OPENAI_API_KEY")

try:
    tavily_key = st.secrets["TAVILY_API_KEY"]
except (KeyError, FileNotFoundError):
    tavily_key = os.environ.get("TAVILY_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set. Please add it to your Streamlit secrets or environment variables.")
    st.stop()

if not tavily_key:
    st.error("TAVILY_API_KEY is not set. Please add it to your Streamlit secrets or environment variables.")
    st.stop()

# Set environment variable for OpenAI SDK
os.environ["OPENAI_API_KEY"] = api_key

@function_tool
def tavily_search(query: str) -> str:
    """Search the internet using Tavily API for current information."""
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=query, search_depth="basic")

    results = response.get('results', [])
    summary = "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results])
    return summary

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = Agent(
    name="Chat Assistant",
    model="gpt-4.1-mini",
    instructions="You are a helpful and friendly assistant. Respond concisely to user queries. Use the tavily_search tool to search the internet for current information when needed.",
    tools=[tavily_search],
)

st.title("A Simple Chatbot Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    async def get_response():
        result = await Runner.run(agent, input=st.session_state.messages)
        return result

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(get_response())

    assistant_response = result.final_output
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    st.rerun()
if st.button("New Chat"):
    st.write("Starting a new chat...")
    st.session_state.messages = []
    st.rerun()
