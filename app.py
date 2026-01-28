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
MCP_TOKEN = st.secrets.get("MCP_TOKEN") or os.getenv("MCP_TOKEN")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not MCP_TOKEN or not GOOGLE_API_KEY:
    st.error("Missing Secrets! Ensure both MCP_TOKEN and GOOGLE_API_KEY are in Streamlit Secrets.")
    st.stop()

SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": "https://anitracker.fastmcp.app/mcp", 
        "headers": {"Authorization": f"Bearer {MCP_TOKEN}"}
    }
}

SYSTEM_PROMPT = "You are a helpful assistant with tool access. Be concise."

# ─────────────────────────────
# ROBUST ASYNC RUNNER
# ─────────────────────────────
def run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ─────────────────────────────
# INITIALIZATION (Cached)
# ─────────────────────────────
@st.cache_resource
def get_mcp_resources():
    client = MultiServerMCPClient(SERVERS)
    try:
        tools = run_sync(client.get_tools())
        
        # FIX: convert_system_message_to_human=True is required for Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
        llm_with_tools = llm.bind_tools(tools)
        
        return client, tools, llm, llm_with_tools
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()

# ─────────────────────────────
# UI & SESSION STATE
# ─────────────────────────────
st.set_page_config(page_title="AniTracker MCP", layout="centered")
st.title("🧰 AniTracker — MCP Chat")

client, tools, llm, llm_with_tools = get_mcp_resources()

if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.tool_by_name = {t.name: t for t in tools}

# Render history (skip system message in UI)
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
        # pass history to llm
        response = run_sync(llm_with_tools.ainvoke(st.session_state.history))
        st.session_state.history.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tool = st.session_state.tool_by_name[tc["name"]]
                result = run_sync(tool.ainvoke(tc["args"]))
                st.session_state.history.append(
                    ToolMessage(tool_call_id=tc["id"], content=json.dumps(result))
                )
            
            # Final LLM pass after tools
            response = run_sync(llm.ainvoke(st.session_state.history))
            st.session_state.history.append(response)

        st.chat_message("assistant").write(response.content)