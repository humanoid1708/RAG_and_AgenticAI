import asyncio
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# MCP Server Configuration
MCP_SERVERS = {
    "context7": {
        "url": "https://mcp.context7.com/mcp",
        "transport": "streamable_http",
    },
    "met-museum": {
        "command": "npx",
        "args": ["-y", "metmuseum-mcp"],
        "transport": "stdio",
    },
}

async def main():
    print("\n==========================================")
    print("       MULTI-SERVER MCP AI AGENT")
    print("==========================================\n")

    client = MultiServerMCPClient(MCP_SERVERS)

    print("Connecting to MCP servers...\n")
    tools = await client.get_tools()

    print("MCP tools loaded:\n-----------------")
    for tool in tools:
        print(f"- {tool.name}")
    print()

    model = ChatOllama(model="llama3.1:8b", temperature=0)
    checkpointer = InMemorySaver()

    agent = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)

    system_message = """
You are a helpful AI assistant with access to multiple MCP servers.

You can use the available MCP tools to:
- Search programming documentation using Context7.
- Search information from the Metropolitan Museum of Art.

Use the appropriate MCP tool when the user's question requires external information.
If a tool is not required, answer normally.
Always provide a clear and concise final answer.
"""

    config = {"configurable": {"thread_id": "mcp-conversation"}}

    print("AI Agent is ready.\nType '2' to exit.\n")

    response = await agent.ainvoke(
        {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Introduce yourself and briefly explain what MCP tools you can use."},
            ]
        },
        config=config,
    )

    print("\nAgent:")
    print(response["messages"][-1].content)

    while True:
        print("\n==========================================")
        print("1. Ask the agent a question")
        print("2. Quit")
        print("==========================================")

        choice = input("\nEnter your choice: ").strip()

        if choice == "2":
            print("\nGoodbye!")
            break
        elif choice == "1":
            question = input("\nYour question: ").strip()
            if not question:
                print("Please enter a question.")
                continue

            print("\nAgent is thinking...\n")
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            print("Agent:")
            print(response["messages"][-1].content)
        else:
            print("\nInvalid choice. Enter 1 or 2.")

if __name__ == "__main__":
    asyncio.run(main())
