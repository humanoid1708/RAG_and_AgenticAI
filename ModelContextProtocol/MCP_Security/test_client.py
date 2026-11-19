import asyncio
from pathlib import Path
import sys
from mcp import Client, StdioServerParameters

BASE_DIR = Path(__file__).parent
SERVER_FILE = BASE_DIR / "server.py"

server = StdioServerParameters(command=sys.executable, args=[str(SERVER_FILE)])

async def main():
    print("=" * 50)
    print("STARTING MCP CLIENT")
    print("=" * 50)

    async with Client(server) as client:
        print("\nCONNECTED TO MCP SERVER")

        result = await client.list_tools()
        print("\nAVAILABLE TOOLS:")
        for tool in result.tools:
            print(f"- {tool.name}")

        print("\n" + "=" * 50)
        print("TESTING read_file")
        print("=" * 50)

        result = await client.call_tool("read_file", {"filepath": "test.txt"})
        print("\nRESULT:")
        for content in result.content:
            if hasattr(content, "text"):
                print(content.text)

        print("\nMCP TEST COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())
