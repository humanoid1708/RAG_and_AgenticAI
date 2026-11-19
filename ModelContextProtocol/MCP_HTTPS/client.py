import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_URL = "http://127.0.0.1:8000/mcp"

async def main():
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("\n==============================")
            print("CONNECTED TO MCP SERVER")
            print("==============================")

            tools = await session.list_tools()
            print("\nTOOLS:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            resources = await session.list_resources()
            print("\nRESOURCES:")
            for resource in resources.resources:
                print(f"- {resource.uri}")

            templates = await session.list_resource_templates()
            print("\nRESOURCE TEMPLATES:")
            for template in templates.resource_templates:
                print(f"- {template.uri_template}")

            print("\nREADING RESOURCE:")
            resource_result = await session.read_resource("file:///workspace/test.txt")
            print(resource_result)

            prompts = await session.list_prompts()
            print("\nPROMPTS:")
            for prompt in prompts.prompts:
                print(f"- {prompt.name}")

            print("\nRENDERED PROMPT:")
            prompt_result = await session.get_prompt("review_code", arguments={"filename": "test.txt"})
            print(prompt_result)

if __name__ == "__main__":
    asyncio.run(main())
