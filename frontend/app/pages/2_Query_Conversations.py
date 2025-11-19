import streamlit as st
import json
import re
import asyncio
import google.generativeai as genai
from fastmcp.client import Client
from dotenv import load_dotenv
import os

from chat import INITIAL_HISTORY

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MCP_URL = os.getenv("MCP_SERVICE_URL", "http://mcp_database_server:8080/sse")

MODEL_NAME = "gemini-2.5-flash"
MAX_TOOL_ITER = 10


# ---------------------------
# Helpers (mirror backend EXACTLY)
# ---------------------------

def parse_json_response(text):
    """Backend uses greedy JSON regex; mirror exactly."""
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def format_final_answer(payload):
    """EXACT same logic as backend version."""
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("answer", "text", "message", "response"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

        if len(payload) == 1:
            key, value = next(iter(payload.items()))
            if isinstance(value, (str, int, float)):
                return f"{key}: {value}"

    try:
        return json.dumps(payload, indent=2)
    except TypeError:
        return str(payload)


async def call_mcp(tool_name, args):
    """Mirror backend behavior."""
    async with Client(MCP_URL) as client:
        result = await client.call_tool(tool_name, args)
        if result.content:
            try:
                return json.loads(result.content[0].text)
            except json.JSONDecodeError:
                return result.content[0].text
        return None


def init_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def start_new_chat():
    st.session_state.model = init_model()
    st.session_state.chat = st.session_state.model.start_chat(history=INITIAL_HISTORY)
    st.session_state.messages = []


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="MCP Chat", layout="wide")
st.title("🧠 Database Chat (MCP + Gemini)")

# Initialize session
if "chat" not in st.session_state:
    start_new_chat()

if st.button("New Chat"):
    start_new_chat()
    st.rerun()

# Show history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------
# User Input
# ---------------------------

user_message = st.chat_input("Ask anything about your database...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.chat_message("user").write(user_message)

    response = st.session_state.chat.send_message(user_message)

    invalid_attempts = 0

    # ---------------------------
    # Tool loop (EXACT backend logic)
    # ---------------------------
    for _ in range(MAX_TOOL_ITER):

        text = response.text
        parsed = parse_json_response(text)

        # Invalid JSON case
        if not parsed:
            invalid_attempts += 1

            if invalid_attempts >= 3:
                final_msg = "I ran into a formatting issue while processing that request. Please ask again."
                st.chat_message("assistant").write(final_msg)
                st.session_state.messages.append({"role": "assistant", "content": final_msg})
                break

            # Send reminder back to model
            response = st.session_state.chat.send_message(
                "Reminder: respond with valid JSON per the system instructions."
            )
            continue

        # Reset invalid counter
        invalid_attempts = 0

        action = parsed.get("action")

        # ---------------------------
        # Tool call
        # ---------------------------
        if action == "tool_call":
            tool = parsed.get("tool")
            args = parsed.get("args", {})

            if not tool:
                err = {"error": "tool_call requested but no 'tool' provided"}
                st.chat_message("assistant").write(err["error"])
                response = st.session_state.chat.send_message(
                    f"Tool result: {json.dumps(err)}"
                )
                break

            st.chat_message("assistant").write(f"🔧 Calling `{tool}`")

            # Call backend tool
            result = asyncio.run(call_mcp(tool, args))

            # Send backend's exact wrapper format
            wrapper = f"Tool result: {json.dumps(result)}"

            response = st.session_state.chat.send_message(wrapper)
            continue

        # ---------------------------
        # FINAL ANSWER (default case)
        # ---------------------------
        final_text = format_final_answer(parsed)

        st.chat_message("assistant").write(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})
        break
