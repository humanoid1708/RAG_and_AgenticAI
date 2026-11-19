import os
import streamlit as st
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Ensure CPU mode on Windows to avoid CUDA initialization issues
os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cpu")


# 1. Tools
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float | str:
    """Divide a by b."""
    if b == 0:
        return "Error: Cannot divide by zero."
    return a / b


# 2. Local Ollama Model & Agent Setup
llm = ChatOllama(model="llama3.1:8b", temperature=0)
tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

system_prompt = """
You are an AI mathematical assistant.
You have access to mathematical tools: add_numbers, subtract_numbers, multiply_numbers, divide_numbers.

Whenever the user asks you to perform arithmetic, use the appropriate tool instead of calculating the answer yourself.
After the tool returns the result, clearly explain the answer to the user.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)


# 3. Streamlit UI
st.set_page_config(page_title="AI Math Assistant", page_icon="🧮", layout="centered")
st.title("🧮 AI Math Assistant")
st.write("Ask a mathematical question and the local AI agent will select and call the appropriate tool.")

user_query = st.text_input("Ask something:", placeholder="Example: What is 25 multiplied by 4?")

if user_query:
    with st.spinner("Thinking..."):
        try:
            response = agent.invoke({"messages": [("human", user_query)]})
            answer = response["messages"][-1].content
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Something went wrong:\n\n{e}")
