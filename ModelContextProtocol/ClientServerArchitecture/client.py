import asyncio
from pathlib import Path
import sys
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.1:8b"

class MCPClient:
    def __init__(self):
        self.client = None
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # MCP Connection
    async def connect_to_server(self, server_script_path):
        print(f"Connecting to server: {server_script_path}")
        self.client = Client(
            Path(server_script_path),
            mode="legacy",
            elicitation_handler=self.handle_elicitation,
            progress_handler=self.handle_progress,
            message_handler=self.handle_message,
        )
        await self.client.__aenter__()
        print(f"Connected to server: {server_script_path}")

    # Handlers
    async def handle_elicitation(self, context, message, params, response_type):
        print("\nServer is requesting information:")
        print(response_type)
        file_path = input("Enter file path: ")
        name = input("Enter document name: ")
        return ElicitResult(action="accept", content={"file_path": file_path, "name": name})

    async def handle_progress(self, progress_token, progress, total, message):
        if total:
            print(f"Progress: {progress}/{total} - {message or ''}")
        else:
            print(f"Progress: {progress} - {message or ''}")

    async def handle_message(self, message):
        print(f"Server message: {message}")

    # MCP Information
    async def get_tools(self):
        return await self.client.list_tools()

    async def get_prompts(self):
        return await self.client.list_prompts()

    async def get_resources(self):
        return await self.client.list_resources()

    async def get_resource_templates(self):
        return await self.client.list_resource_templates()

    # Convert MCP Tools -> LangChain Tools
    async def get_langchain_tools(self):
        tools = await self.get_tools()
        langchain_tools = []

        for tool in tools:
            try:
                input_schema = tool.input_schema
            except AttributeError:
                input_schema = tool.inputSchema

            def create_tool(tool_name, schema):
                async def mcp_tool(**kwargs):
                    result = await self.client.call_tool(tool_name, kwargs)
                    return str(result)

                mcp_tool.__name__ = tool_name
                mcp_tool.__doc__ = f"MCP tool: {tool_name}"
                return mcp_tool

            tool_function = create_tool(tool.name, input_schema)
            langchain_tools.append(tool_function)

        return langchain_tools

    # Simple Ollama Conversation
    async def converse(self):
        print("\nEntering conversation mode.\nType 'quit' or 'q' to exit.\n")
        while True:
            query = input("Query: ")
            if query.lower() in ["quit", "q"]:
                break
            try:
                response = await self.llm.ainvoke([HumanMessage(content=query)])
                print(f"\nAssistant:\n{response.content}\n")
            except Exception as e:
                print(f"Error processing query: {e}")

    # Run MCP Prompt
    async def run_prompt(self, prompt_name, arguments=None):
        try:
            result = await self.client.get_prompt(prompt_name, arguments=arguments or {})
            messages = []
            for message in result.messages:
                if hasattr(message, "content"):
                    content = message.content
                    if isinstance(content, str):
                        messages.append(HumanMessage(content=content))
                    elif hasattr(content, "text"):
                        messages.append(HumanMessage(content=content.text))

            if not messages:
                print("Prompt returned no usable messages.")
                return

            response = await self.llm.ainvoke(messages)
            print(f"\nAssistant:\n{response.content}")
        except Exception as e:
            print(f"Error executing prompt: {e}")

    # Read File Resource
    async def read_file(self):
        file_path = input("Enter file name: ")
        try:
            result = await self.client.read_resource(f"file:///{file_path}")
            print(f"\nFile contents:\n{result}")
        except Exception as e:
            print(f"Error reading file: {e}")

    # Read Directory Resource
    async def read_dir(self):
        try:
            result = await self.client.read_resource("dir:///")
            print(f"\nCurrent Directory:\n{result}")
        except Exception as e:
            print(f"Error reading directory: {e}")

    # Menu
    async def menu(self):
        while True:
            print("""
Select from the Menu:
1. Generate Documentation
2. Review Code
3. Read File
4. Read Current Directory
5. Converse with Agent
q. Quit
""")
            choice = input("Choice: ").strip()
            if choice == "1":
                await self.run_prompt("documentation_generator")
            elif choice == "2":
                file_path = input("Enter file_path (required): ")
                await self.run_prompt("code_review", {"file_path": file_path})
            elif choice == "3":
                await self.read_file()
            elif choice == "4":
                await self.read_dir()
            elif choice == "5":
                await self.converse()
            elif choice.lower() == "q":
                print("Exiting client...")
                break
            else:
                print("Invalid choice.")

    # Cleanup
    async def cleanup(self):
        if self.client:
            await self.client.__aexit__(None, None, None)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py server.py")
        return

    server_script_path = sys.argv[1]
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        print("\nMCP Client Started")
        await client.menu()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
