import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import streamlit as st

SERVER_URL = "http://127.0.0.1:8000/mcp"

# MCP Connection Helpers
async def get_tools():
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools

async def call_tool(tool_name, arguments):
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            output = [item.text for item in result.content if hasattr(item, "text")]
            return "\n".join(output)

async def list_resources():
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_resource_templates()
            return result.resourceTemplates

async def read_resource(uri):
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            output = [item.text for item in result.contents if hasattr(item, "text")]
            return "\n".join(output)

async def list_prompts():
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_prompts()
            return result.prompts

async def get_prompt(prompt_name, filename):
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.get_prompt(prompt_name, arguments={"filename": filename})
            output = [f"{msg.role}: {msg.content.text}" for msg in result.messages if hasattr(msg.content, "text")]
            return "\n\n".join(output)

# Streamlit UI
st.set_page_config(page_title="MCP HTTP Lab", page_icon="🔌", layout="wide")
st.title("MCP HTTP Client")
st.write("Streamlit → HTTP → MCP Server → Workspace")
st.caption("Tools • Resources • Prompts • Roots Security")

st.header("MCP Server")
if st.button("Connect & Discover Server", type="primary"):
    try:
        tools = asyncio.run(get_tools())
        st.session_state["tools"] = tools
        st.session_state["connected"] = True
        st.success("Successfully connected to MCP server.")

        try:
            resources = asyncio.run(list_resources())
            st.session_state["resources"] = resources
        except Exception as e:
            st.session_state["resources"] = []
            st.warning(f"Resources could not be discovered: {e}")

        try:
            prompts = asyncio.run(list_prompts())
            st.session_state["prompts"] = prompts
        except Exception as e:
            st.session_state["prompts"] = []
            st.warning(f"Prompts could not be discovered: {e}")

    except Exception as e:
        st.session_state["connected"] = False
        st.error(f"Could not connect to MCP server: {e}")

if st.session_state.get("connected", False):
    tools = st.session_state.get("tools", [])
    resources = st.session_state.get("resources", [])
    prompts = st.session_state.get("prompts", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tools", len(tools))
    with col2:
        st.metric("Resources", len(resources))
    with col3:
        st.metric("Prompts", len(prompts))

    st.divider()

    tools_tab, resources_tab, prompts_tab = st.tabs(["Tools", "Resources", "Prompts"])

    # Tools Tab
    with tools_tab:
        st.header("MCP Tools")
        st.write("Tools are actions that the MCP server can perform.")

        for tool in tools:
            st.write(f"**{tool.name}**")
            if tool.description:
                st.caption(tool.description)

        st.divider()
        tool_names = [tool.name for tool in tools]

        if tool_names:
            selected_tool = st.selectbox("Select MCP Tool", tool_names)

            if selected_tool == "read_file":
                st.subheader("Read File")
                filepath = st.text_input("File path", "test.txt", key="read_filepath")
                if st.button("Read File", key="read_file_button"):
                    try:
                        result = asyncio.run(call_tool("read_file", {"filepath": filepath}))
                        st.code(result)
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif selected_tool == "write_file":
                st.subheader("Write File")
                filepath = st.text_input("File path", "new.txt", key="write_filepath")
                content = st.text_area("File content", "Hello from Streamlit!", key="write_content")
                if st.button("Write File", key="write_file_button"):
                    try:
                        result = asyncio.run(call_tool("write_file", {"filepath": filepath, "content": content}))
                        st.success(result)
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif selected_tool == "list_files":
                st.subheader("List Files")
                directory = st.text_input("Directory", ".", key="list_directory")
                if st.button("List Files", key="list_files_button"):
                    try:
                        result = asyncio.run(call_tool("list_files", {"directory": directory}))
                        st.code(result)
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif selected_tool == "analyze_code":
                st.subheader("Analyze Code")
                code = st.text_area("Code", "def add(a, b):\n    return a + b", height=200, key="analysis_code")
                focus = st.selectbox("Analysis Focus", ["quality", "security", "performance"], key="analysis_focus")
                if st.button("Analyze Code", key="analyze_button"):
                    try:
                        result = asyncio.run(call_tool("analyze_code", {"code": code, "focus": focus}))
                        st.code(result)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Resources Tab
    with resources_tab:
        st.header("MCP Resources")
        st.write("Resources provide read-only data from the MCP server.")

        if resources:
            st.subheader("Available Resource Templates")
            for resource in resources:
                uri_template = getattr(resource, "uriTemplate", None)
                if uri_template:
                    st.code(uri_template)
        else:
            st.info("No resource templates available.")

        st.divider()
        st.subheader("Read Resource")
        resource_filename = st.text_input("Resource filename", "test.txt", key="resource_filename")
        resource_uri = f"file:///workspace/{resource_filename}"
        st.write("Resource URI:")
        st.code(resource_uri)

        if st.button("Read Resource", key="read_resource_button"):
            try:
                result = asyncio.run(read_resource(resource_uri))
                st.code(result)
            except Exception as e:
                st.error(f"Error: {e}")

    # Prompts Tab
    with prompts_tab:
        st.header("MCP Prompts")
        st.write("Prompts are reusable instruction templates exposed by the MCP server.")

        prompt_names = []
        if prompts:
            st.subheader("Available Prompts")
            for prompt in prompts:
                prompt_names.append(prompt.name)
                st.write(f"**{prompt.name}**")
                if prompt.description:
                    st.caption(prompt.description)
        else:
            st.info("No prompts available.")

        st.divider()
        if prompt_names:
            st.subheader("Get Prompt")
            selected_prompt = st.selectbox("Select Prompt", prompt_names, key="selected_prompt")
            prompt_filename = st.text_input("Filename", "test.txt", key="prompt_filename")
            if st.button("Get Prompt", key="get_prompt_button"):
                try:
                    result = asyncio.run(get_prompt(selected_prompt, prompt_filename))
                    st.code(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Roots / Security
    st.divider()
    st.header("Roots / Security")
    st.info("""
All filesystem operations are restricted to the workspace directory configured on the MCP server.

Try:
../../etc/passwd
or:
../server.py

The MCP server should reject these paths.
""")
