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
# CONFIGURATION
# ─────────────────────────────
# 1. Get secrets securely
MCP_TOKEN = st.secrets.get("MCP_TOKEN") or os.getenv("MCP_TOKEN")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not MCP_TOKEN or not GOOGLE_API_KEY:
    st.error("🚨 Missing Secrets! Please add `MCP_TOKEN` and `GOOGLE_API_KEY` to your Streamlit Secrets.")
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
# SYNC RUNNER (Fixes Async Errors)
# ─────────────────────────────
def run_sync(coro):
    """Safely runs async code in Streamlit's threaded environment."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ─────────────────────────────
# INITIALIZATION
# ─────────────────────────────
@st.cache_resource
def get_mcp_resources():
    client = MultiServerMCPClient(SERVERS)
    try:
        # Check connection to MCP server
        tools = run_sync(client.get_tools())
        
        # Use stable Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
        llm_with_tools = llm.bind_tools(tools)
        
        return client, tools, llm, llm_with_tools
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        st.stop()

st.set_page_config(page_title="AniTracker MCP", layout="centered")
st.title("🧰 AniTracker — MCP Chat")

# Initialize Client
client, tools, llm, llm_with_tools = get_mcp_resources()

# Session State
if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.tool_by_name = {t.name: t for t in tools}

# Render Chat
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# ─────────────────────────────
# MAIN CHAT LOOP
# ─────────────────────────────
user_text = st.chat_input("Ask about expenses...")

if user_text:
    st.chat_message("user").write(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    with st.spinner("Executing..."):
        try:
            # 1. First LLM Call
            response = run_sync(llm_with_tools.ainvoke(st.session_state.history))
            st.session_state.history.append(response)

            # 2. Tool Execution (if needed)
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool = st.session_state.tool_by_name.get(tc["name"])
                    if tool:
                        result = run_sync(tool.ainvoke(tc["args"]))
                        st.session_state.history.append(
                            ToolMessage(tool_call_id=tc["id"], content=json.dumps(result))
                        )
                
                # 3. Final Answer
                response = run_sync(llm.ainvoke(st.session_state.history))
                st.session_state.history.append(response)

            st.chat_message("assistant").write(response.content)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")