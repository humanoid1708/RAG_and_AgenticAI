import asyncio
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import ollama
import streamlit as st

# Configuration
SERVER_URL = "http://127.0.0.1:8000/mcp"
MODEL = "llama3.1:8b"

# MCP Tool Helpers
async def get_mcp_tools():
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools

async def call_mcp_tool(tool_name, arguments):
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            output = [item.text for item in result.content if hasattr(item, "text")]
            return "\n".join(output)

def convert_mcp_tools_to_ollama(mcp_tools):
    ollama_tools = []
    for tool in mcp_tools:
        input_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}

        if not isinstance(input_schema, dict):
            try:
                input_schema = input_schema.model_dump(by_alias=True)
            except Exception:
                input_schema = {"type": "object", "properties": {}}

        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"Execute MCP tool {tool.name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return ollama_tools

def execute_tool_call(tool_name, arguments):
    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return asyncio.run(call_mcp_tool(tool_name, arguments))
    except Exception as e:
        return f"Error executing MCP tool '{tool_name}': {e}"

def chat_with_agent(user_message):
    mcp_tools = asyncio.run(get_mcp_tools())
    ollama_tools = convert_mcp_tools_to_ollama(mcp_tools)

    system_message = {
        "role": "system",
        "content": """
You are an AI assistant connected to an MCP server.
The MCP server provides tools for working with a restricted workspace.

Available operations include:
- Reading files
- Writing files
- Listing files
- Analyzing code

Rules:
1. Use MCP tools whenever the user asks you to read, write, list, or analyze workspace files.
2. Never claim that a file was created, modified, read, or analyzed unless the MCP tool actually returned a result.
3. Never directly access the filesystem.
4. Always use the MCP tools for workspace operations.
5. Respect security restrictions imposed by the MCP server.
6. If an MCP tool returns an error, explain the error to the user.
7. After using a tool, provide a concise natural language response.
""",
    }

    messages = [system_message, {"role": "user", "content": user_message}]

    while True:
        response = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools)
        assistant_message = response["message"]
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls:
            return assistant_message.get("content", "No response generated.")

        for tool_call in tool_calls:
            function = tool_call["function"]
            tool_name = function["name"]
            arguments = function.get("arguments", {})

            with st.status(f"Executing MCP tool: `{tool_name}`", expanded=True):
                st.write("Arguments:")
                st.json(arguments)
                tool_result = execute_tool_call(tool_name, arguments)
                st.write("MCP Result:")
                st.code(tool_result)

            messages.append({"role": "tool", "content": tool_result})

# Streamlit Configuration
st.set_page_config(page_title="MCP AI Host", page_icon="🤖", layout="centered")
st.title("MCP AI Host")
st.write("Ollama + MCP HTTP Server")
st.caption(f"Local LLM: {MODEL}")

with st.expander("View Architecture"):
    st.code("""
User → Streamlit → Ollama → Tool Selection → MCP Client → Streamable HTTP → MCP Server → Workspace
""", language="text")

with st.sidebar:
    st.header("MCP AI Host")
    st.write(f"Server: `{SERVER_URL}`")
    st.write(f"Model: `{MODEL}`")
    st.divider()

    if st.button("Test MCP Connection"):
        try:
            tools = asyncio.run(get_mcp_tools())
            st.success("Connected to MCP server.")
            st.write("Available tools:")
            for tool in tools:
                st.write(f"- `{tool.name}`")
        except Exception as e:
            st.error(str(e))

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_message = st.chat_input("Ask the MCP AI assistant something...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking and using MCP tools..."):
                response = chat_with_agent(user_message)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error communicating with MCP server or Ollama: {e}")
