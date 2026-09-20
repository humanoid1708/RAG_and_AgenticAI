from langchain_ollama import ChatOllama
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.1:8b")

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a+b

llm_tools = llm.bind_tools([add])

response = llm_tools.invoke(
    "What is 4+8?"
)

print(response)
print(response.tool_calls)


'''
What is Tool Binding?

We give tools to the LLM:

llm_with_tools = llm.bind_tools(
    [add, subtract, multiply]
)

Now the LLM knows:

I have access to:

1. add()
2. subtract()
3. multiply()

and can decide which one should be called.
'''