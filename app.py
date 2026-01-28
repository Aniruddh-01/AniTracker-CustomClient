# app.py — MCP + Streamlit chat using Google Gemini

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

# ─────────────────────────────
# MCP servers
# ─────────────────────────────
SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": "https://anitracker.fastmcp.app/mcp",
    }
}

SYSTEM_PROMPT = (
    "You have access to tools.\n"
    "When you choose to call a tool, do not narrate status updates.\n"
    "After tools run, return only a concise final answer."
)

st.set_page_config(
    page_title="MCP Chat",
    page_icon="🧰",
    layout="centered",
)

st.title("🧰 AniTracker — MCP Chat")

load_dotenv()

# ─────────────────────────────
# One-time initialization
# ─────────────────────────────
if "initialized" not in st.session_state:

    # ✅ Google Gemini LLM
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    # ✅ MCP Client
    st.session_state.client = MultiServerMCPClient(SERVERS)

    tools = asyncio.run(
        st.session_state.client.get_tools()
    )

    st.session_state.tools = tools
    st.session_state.tool_by_name = {
        t.name: t for t in tools
    }

    # ✅ Bind tools to Gemini
    st.session_state.llm_with_tools = (
        st.session_state.llm.bind_tools(tools)
    )

    # Conversation memory
    st.session_state.history = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    st.session_state.initialized = True


# ─────────────────────────────
# Render chat history
# ─────────────────────────────
for msg in st.session_state.history:

    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        # hide tool-call planning messages
        if getattr(msg, "tool_calls", None):
            continue

        with st.chat_message("assistant"):
            st.markdown(msg.content)


# ─────────────────────────────
# Chat input
# ─────────────────────────────
user_text = st.chat_input("Ask about expenses…")

if user_text:

    with st.chat_message("user"):
        st.markdown(user_text)

    st.session_state.history.append(
        HumanMessage(content=user_text)
    )

    # ── First model pass ──
    first = asyncio.run(
        st.session_state.llm_with_tools.ainvoke(
            st.session_state.history
        )
    )

    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # normal text response
        with st.chat_message("assistant"):
            st.markdown(first.content or "")

        st.session_state.history.append(first)

    else:
        # 1️⃣ store tool call message (hidden)
        st.session_state.history.append(first)

        # 2️⃣ execute tools
        tool_msgs = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            tool = st.session_state.tool_by_name[name]

            result = asyncio.run(
                tool.ainvoke(args)
            )

            tool_msgs.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    content=json.dumps(result),
                )
            )

        st.session_state.history.extend(tool_msgs)

        # 3️⃣ final response
        final = asyncio.run(
            st.session_state.llm.ainvoke(
                st.session_state.history
            )
        )

        with st.chat_message("assistant"):
            st.markdown(final.content or "")

        st.session_state.history.append(
            AIMessage(content=final.content or "")
        )
