from langchain_core.tools import tool
from langchain_ollama import ChatOllama

@tool
def add(a: int, b: int) -> int:
    """Adding two numbers"""
    return a+b

@tool
def sub(a: int, b: int) -> int:
    """Subtracting two numbers"""
    return a-b

llm = ChatOllama(model="llama3.1:8b")

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assisstant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")

])

agent = create_tool_calling_agent(
    llm,
    [add, sub],
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[add, sub],
    verbose=True
)

response = agent_executor.invoke(
    {
        "input": "What is 25 added to 12?"
    }
)

print(response["output"])