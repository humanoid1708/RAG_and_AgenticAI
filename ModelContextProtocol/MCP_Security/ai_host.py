import ollama
import streamlit as st
from client import MCPPermissionClient

st.set_page_config(page_title="MCP AI Security Host", page_icon="🤖", layout="centered")

MODEL = "llama3.1:8b"
client = MCPPermissionClient()

TOOL_DESCRIPTIONS = {
    "read_file": {"description": "Read a file", "risk": "low"},
    "write_file": {"description": "Write content to a file", "risk": "medium"},
    "delete_file": {"description": "Delete a file", "risk": "high"},
    "execute_command": {"description": "Simulate execution of a system command", "risk": "critical"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Simulate a system command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None

st.title("MCP Security AI Host")
st.caption("Ollama + MCP + Permission Enforcement")

with st.expander("Permission Status"):
    for tool in TOOL_DESCRIPTIONS:
        permission = client.get_permission(tool)
        risk = client.get_risk(tool)
        st.write(f"**{tool}** → Permission: `{permission.upper()}` | Risk: `{risk.upper()}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.pending_tool:
    pending = st.session_state.pending_tool
    st.warning(f"Approval required\n\nTool: `{pending['tool']}`\n\nRisk: `{pending['risk'].upper()}`\n\nArguments:")
    st.json(pending["arguments"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve"):
            result = client.call_tool(pending["tool"], pending["arguments"], approval=True)
            st.session_state.messages.append({"role": "assistant", "content": result["message"]})
            st.session_state.pending_tool = None
            st.rerun()
    with col2:
        if st.button("Reject"):
            client.log("REJECTED", pending["tool"], "User rejected approval")
            st.session_state.messages.append({"role": "assistant", "content": "Operation rejected by user."})
            st.session_state.pending_tool = None
            st.rerun()

user_input = st.chat_input("Ask the AI to perform an operation...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    system_message = """
You are a security-aware MCP assistant.
You have access to these tools:
- read_file: Reads files (LOW risk)
- write_file: Writes files (MEDIUM risk)
- delete_file: Deletes files (HIGH risk)
- execute_command: Simulates system commands (CRITICAL risk)

Always use the appropriate tool.
Never claim an operation succeeded unless the tool actually returned success.
The application will enforce permissions.
"""

    messages = [{"role": "system", "content": system_message}]
    messages.extend(st.session_state.messages)

    try:
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        assistant_message = response["message"]

        if assistant_message.get("tool_calls"):
            tool_call = assistant_message["tool_calls"][0]
            function = tool_call["function"]
            tool_name = function["name"]
            arguments = function["arguments"]

            risk = client.get_risk(tool_name)
            permission = client.get_permission(tool_name)

            st.info(f"Tool: `{tool_name}`\n\nRisk: `{risk.upper()}`\n\nPermission: `{permission.upper()}`")
            result = client.call_tool(tool_name, arguments)

            if result["status"] == "approval_required":
                st.session_state.pending_tool = {"tool": tool_name, "arguments": arguments, "risk": risk}
                st.rerun()
            elif result["status"] == "denied":
                final_message = f"Operation blocked.\n\n{result['message']}"
                st.session_state.messages.append({"role": "assistant", "content": final_message})
                with st.chat_message("assistant"):
                    st.error(final_message)
            elif result["status"] == "success":
                final_message = f"Operation completed.\n\n{result['message']}"
                st.session_state.messages.append({"role": "assistant", "content": final_message})
                with st.chat_message("assistant"):
                    st.success(final_message)
            else:
                st.session_state.messages.append({"role": "assistant", "content": result["message"]})
                with st.chat_message("assistant"):
                    st.error(result["message"])

        else:
            content = assistant_message.get("content", "I could not generate a response.")
            st.session_state.messages.append({"role": "assistant", "content": content})
            with st.chat_message("assistant"):
                st.write(content)

    except Exception as e:
        st.error(f"AI error: {str(e)}")
