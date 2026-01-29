import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI 
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
MCP_TOKEN = st.secrets.get("MCP_TOKEN") or os.getenv("MCP_TOKEN")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not MCP_TOKEN or not OPENAI_API_KEY:
    st.error("🚨 Missing Secrets! Please add `OPENAI_API_KEY` and `MCP_TOKEN` to your settings.")
    st.stop()

SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": "https://aniruddh.fastmcp.app/mcp", 
        "headers": {"Authorization": f"Bearer {MCP_TOKEN}"}
    }
}

# Added instruction to avoid JSON in the final output
SYSTEM_PROMPT = (
    "You are a helpful assistant with tool access. "
    "When providing final answers after using a tool, always summarize the data "
    "in a clear, human-readable textual format. Never show raw JSON to the user."
)

# ─────────────────────────────
# SYNC RUNNER
# ─────────────────────────────
def run_sync(coro):
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
        tools = run_sync(client.get_tools())
        
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)
        
        return client, tools, llm, llm_with_tools
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        st.stop()

st.set_page_config(page_title="AniTracker MCP", layout="centered")
st.title("🧰 AniTracker — OpenAI Edition")

client, tools, llm, llm_with_tools = get_mcp_resources()

if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.tool_by_name = {t.name: t for t in tools}

# Render Chat History
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        # This ensures we only render the final text content, not the tool_calls metadata
        st.chat_message("assistant").write(msg.content)

# ─────────────────────────────
# MAIN CHAT LOOP
# ─────────────────────────────
user_text = st.chat_input("Ask about expenses...")

if user_text:
    st.chat_message("user").write(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    with st.spinner("Processing..."):
        try:
            # Step 1: LLM decides if a tool is needed
            response = llm_with_tools.invoke(st.session_state.history)
            st.session_state.history.append(response)

            # Step 2: If tool calls exist, execute them and get a final text response
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool = st.session_state.tool_by_name.get(tc["name"])
                    if tool:
                        result = run_sync(tool.ainvoke(tc["args"]))
                        # Add the raw tool output to history (not displayed to user)
                        st.session_state.history.append(
                            ToolMessage(tool_call_id=tc["id"], content=json.dumps(result))
                        )
                
                # Step 3: Final LLM call to turn JSON tool results into Text
                final_response = llm.invoke(st.session_state.history)
                st.session_state.history.append(final_response)
                st.chat_message("assistant").write(final_response.content)
            else:
                # If no tool was needed, just print the direct response
                st.chat_message("assistant").write(response.content)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")