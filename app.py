import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
# 1. Pull the token from Streamlit Secrets (for Cloud) or .env (for Local)
MCP_TOKEN = st.secrets.get("MCP_TOKEN") or os.getenv("MCP_TOKEN")

if not MCP_TOKEN:
    st.error("Missing MCP_TOKEN! Please add it to your Streamlit Secrets or .env file.")
    st.stop()

SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": "https://anitracker.fastmcp.app/mcp", 
        "headers": {
            "Authorization": f"Bearer {MCP_TOKEN}"
        }
    }
}

SYSTEM_PROMPT = "You are a helpful assistant with tool access. Be concise."

# ─────────────────────────────
# ROBUST ASYNC RUNNER (Python 3.13 Compatible)
# ─────────────────────────────
def run_sync(coro):
    """
    Safely runs async code in Streamlit's threaded environment.
    Fixes: 'RuntimeError: There is no current event loop in thread'
    """
    try:
        # Check if there is already a running loop in this thread
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If not, create a new one and set it as the thread's loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# ─────────────────────────────
# INITIALIZATION (Cached)
# ─────────────────────────────
@st.cache_resource
def get_mcp_resources():
    """Connects to MCP and LLM once and persists them."""
    client = MultiServerMCPClient(SERVERS)
    try:
        # Fetch available tools from the server
        tools = run_sync(client.get_tools())
        
        # Initialize Gemini 2.0 Flash
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        
        return client, tools, llm, llm_with_tools
    
    except Exception as e:
        # Error unwrapper for ExceptionGroups (common in async)
        if hasattr(e, "exceptions"):
            for i, sub_e in enumerate(e.exceptions):
                st.error(f"Root Cause {i+1}: {sub_e}")
        else:
            st.error(f"Connection Error: {e}")
        st.stop()

# ─────────────────────────────
# UI & SESSION STATE
# ─────────────────────────────
st.set_page_config(page_title="AniTracker MCP", layout="centered")
st.title("🧰 AniTracker — MCP Chat")

# Load persistent client
client, tools, llm, llm_with_tools = get_mcp_resources()

if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.tool_by_name = {t.name: t for t in tools}

# Render chat history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# ─────────────────────────────
# CHAT INPUT
# ─────────────────────────────
user_text = st.chat_input("Ask about expenses...")

if user_text:
    st.chat_message("user").write(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    with st.spinner("Executing..."):
        # 1. LLM decides whether to use a tool
        response = run_sync(llm_with_tools.ainvoke(st.session_state.history))
        st.session_state.history.append(response)

        # 2. If tool calls are requested, execute them
        if response.tool_calls:
            for tc in response.tool_calls:
                tool = st.session_state.tool_by_name[tc["name"]]
                result = run_sync(tool.ainvoke(tc["args"]))
                
                # Append the result of the tool to history
                st.session_state.history.append(
                    ToolMessage(tool_call_id=tc["id"], content=json.dumps(result))
                )
            
            # 3. Get the final response from LLM using the tool data
            response = run_sync(llm.ainvoke(st.session_state.history))
            st.session_state.history.append(response)

        # Display assistant message
        st.chat_message("assistant").write(response.content)