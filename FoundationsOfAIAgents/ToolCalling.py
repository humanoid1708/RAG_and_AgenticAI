import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. TOOLS
# ============================================================

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
def divide_numbers(a: float, b: float) -> float:
    """Divide a by b."""

    if b == 0:
        return "Error: Cannot divide by zero."

    return a / b


# ============================================================
# 2. LOCAL OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


# ============================================================
# 3. GIVE TOOLS TO THE AGENT
# ============================================================

tools = [
    add_numbers,
    subtract_numbers,
    multiply_numbers,
    divide_numbers
]


# ============================================================
# 4. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an AI mathematical assistant.

You have access to mathematical tools.

Whenever the user asks you to perform arithmetic,
use the appropriate tool instead of calculating the
answer yourself.

Available operations:
- Addition
- Subtraction
- Multiplication
- Division

After the tool returns the result, clearly explain
the answer to the user.

For example:

User: What is 25 multiplied by 4?

Use:
multiply_numbers(25, 4)

Then answer:
25 × 4 = 100
"""
)


# ============================================================
# 5. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Math Assistant",
    page_icon="🧮",
    layout="centered"
)


# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.title("🧮 AI Math Assistant")

st.write(
    "Ask a mathematical question and the local AI "
    "will choose the appropriate tool."
)


# ============================================================
# 7. USER INPUT
# ============================================================

user_query = st.text_input(
    "Ask something:",
    placeholder="Example: What is 25 multiplied by 4?"
)


# ============================================================
# 8. RUN AGENT
# ============================================================

if user_query:

    with st.spinner("Thinking..."):

        try:

            response = agent.invoke(
                {
                    "messages": [
                        ("human", user_query)
                    ]
                }
            )

            # Get the final AI response
            answer = response["messages"][-1].content

            st.subheader("Answer")

            st.write(answer)

        except Exception as e:

            st.error(
                f"Something went wrong:\n\n{str(e)}"
            )