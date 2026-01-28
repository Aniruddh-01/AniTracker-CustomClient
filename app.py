# app.py — Robust MCP + Streamlit chat
import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

load_dotenv()

# ─────────────────────────────
# CONFIG & ASYNC HELPER
# ─────────────────────────────
SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": "https://anitracker.fastmcp.app/mcp",
    }
}

SYSTEM_PROMPT = (
    "You are a helpful expense tracking assistant. You have access to tools.\n"
    "When calling a tool, do not narrate status updates. "
    "Return only a concise final answer after the tool runs."
)

def run_sync(coro):
    """Safely run coroutines in Streamlit's environment."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback if a loop is already running in this thread
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# ─────────────────────────────
# CACHED INITIALIZATION
# ─────────────────────────────
@st.cache_resource
def initialize_mcp_and_llm():
    """Persistent connection to MCP and LLM."""
    try:
        # 1. Initialize Client
        client = MultiServerMCPClient(SERVERS)
        
        # 2. Fetch Tools (This is where your error was happening)
        # Wrapping in a try/except to catch the ExceptionGroup
        tools = run_sync(client.get_tools())
        
        # 3. Setup LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Use "gemini-1.5-flash" if 2.5 is unavailable
            temperature=0,
        )
        llm_with_tools = llm.bind_tools(tools)
        
        return client, tools, llm, llm_with_tools
    except Exception as e:
        st.error("Failed to connect to the MCP Server.")
        st.exception(e) # Reveals the 'hidden' cause inside the ExceptionGroup
        st.stop()

# ─────────────────────────────
# UI SETUP
# ─────────────────────────────
st.set_page_config(page_title="AniTracker MCP", page_icon="🧰")
st.title("🧰 AniTracker — MCP Chat")

# Initialize persistent components
client, tools, llm, llm_with_tools = initialize_mcp_and_llm()

if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.tool_by_name = {t.name: t for t in tools}

# ─────────────────────────────
# CHAT LOGIC
# ─────────────────────────────
# Display existing history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and not msg.tool_calls:
        st.chat_message("assistant").write(msg.content)

# Input
user_text = st.chat_input("Ask about expenses...")

if user_text:
    st.chat_message("user").write(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    with st.spinner("Thinking..."):
        # Pass 1: LLM decides if tools are needed
        response = run_sync(llm_with_tools.ainvoke(st.session_state.history))
        
        if not response.tool_calls:
            st.chat_message("assistant").write(response.content)
            st.session_state.history.append(response)
        else:
            # Handle Tool Calls
            st.session_state.history.append(response)
            
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                
                # Execute Tool
                tool = st.session_state.tool_by_name[name]
                tool_result = run_sync(tool.ainvoke(args))
                
                st.session_state.history.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=json.dumps(tool_result),
                    )
                )
            
            # Pass 2: Final response with tool data
            final_response = run_sync(llm.ainvoke(st.session_state.history))
            st.chat_message("assistant").write(final_response.content)
            st.session_state.history.append(final_response)