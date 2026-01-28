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
# NOTE: If your server is running FastMCP, the URL is often just 
# "https://anitracker.fastmcp.app/sse" or the root. 
# Check your deployment logs to confirm the path.
MCP_TOKEN = st.secrets.get("MCP_TOKEN") or os.getenv("MCP_TOKEN")

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
# ROBUST ASYNC RUNNER
# ─────────────────────────────
def run_sync(coro):
    """Safely runs async code even if an event loop is already running."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# ─────────────────────────────
# INITIALIZATION (Cached)
# ─────────────────────────────
@st.cache_resource
def get_mcp_resources():
    """Connects to MCP and LLM once and persists them."""
    client = MultiServerMCPClient(SERVERS)
    try:
        # Try to fetch tools
        tools = run_sync(client.get_tools())
        
        # Using Gemini 2.0 (2.5 is not a valid model string yet)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        
        return client, tools, llm, llm_with_tools
    
    except Exception as e:
        # --- ERROR UNWRAPPER ---
        # This part looks inside the 'ExceptionGroup' to show you the REAL error
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

# Render history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and not msg.tool_calls:
        st.chat_message("assistant").write(msg.content)

# ─────────────────────────────
# CHAT INPUT
# ─────────────────────────────
user_text = st.chat_input("Ask about expenses...")

if user_text:
    st.chat_message("user").write(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    with st.spinner("Executing..."):
        # 1. First Pass
        response = run_sync(llm_with_tools.ainvoke(st.session_state.history))
        st.session_state.history.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tool = st.session_state.tool_by_name[tc["name"]]
                # 2. Tool Pass
                result = run_sync(tool.ainvoke(tc["args"]))
                st.session_state.history.append(
                    ToolMessage(tool_call_id=tc["id"], content=json.dumps(result))
                )
            
            # 3. Final Pass
            response = run_sync(llm.ainvoke(st.session_state.history))
            st.session_state.history.append(response)

        st.chat_message("assistant").write(response.content)