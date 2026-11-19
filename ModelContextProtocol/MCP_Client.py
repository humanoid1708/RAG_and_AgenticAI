import asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

# Configuration
CONTEXT7_URL = "https://mcp.context7.com/mcp"

def create_stdio_client():
    transport = StdioTransport(command="npx", args=["-y", "@upstash/context7-mcp"])
    return Client(transport)

def create_http_client():
    transport = StreamableHttpTransport(url=CONTEXT7_URL)
    return Client(transport)

async def list_tools(client):
    tools = await client.list_tools()
    print("\nAvailable MCP Tools\n-------------------")
    for tool in tools:
        print(f"- {tool.name}")
    return tools

async def resolve_library(client):
    print("\nSearching for FastMCP...")
    result = await client.call_tool(
        "resolve-library-id",
        {
            "libraryName": "fastmcp",
            "query": "I want to create an MCP server using the FastMCP Python framework",
        },
    )
    print("\nLibrary Search Result\n---------------------")
    print(result.content[0].text)
    return result

async def query_documentation(client):
    print("\nFetching FastMCP documentation...")
    library_id = "/llmstxt/gofastmcp_llms-full_txt"
    result = await client.call_tool(
        "query-docs",
        {
            "libraryId": library_id,
            "query": "How do I create an MCP server using FastMCP?",
            "tokens": 5000,
        },
    )
    print("\nDocumentation\n-------------")
    print(result.content[0].text[:5000])
    return result

async def run_client(mode):
    if mode == "stdio":
        print("\nUsing STDIO transport...")
        client = create_stdio_client()
    else:
        print("\nUsing HTTP transport...")
        client = create_http_client()

    async with client:
        print("Connected to Context7 MCP Server.")
        await list_tools(client)
        await resolve_library(client)
        await query_documentation(client)

async def main():
    print("\n======================================")
    print("       MCP EXISTING SERVER LAB")
    print("======================================")
    print("\nChoose connection method:\n1. STDIO\n2. HTTP")

    choice = input("\nEnter your choice: ").strip()
    if choice == "1":
        await run_client("stdio")
    elif choice == "2":
        await run_client("http")
    else:
        print("\nInvalid choice.")

if __name__ == "__main__":
    asyncio.run(main())
