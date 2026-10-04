import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

MAX_ITERATIONS = 3

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    request: str
    post: str
    critique: str
    iteration: int


# ============================================================
# GENERATE NODE
# ============================================================

def generate_node(state: State):

    prompt = f"""
You are a professional LinkedIn content writer.

Create a high-quality LinkedIn post based on the user's request.

User request:
{state["request"]}

Previous post:
{state["post"]}

Previous critique:
{state["critique"]}

If a previous post and critique exist, improve the
previous post using the critique.

Requirements:

- Professional tone
- Clear and engaging
- Easy to read
- Strong opening
- Useful content
- Appropriate LinkedIn style
- Do not mention AI
- Do not mention the reflection process

Return ONLY the LinkedIn post.
"""

    response = llm.invoke(prompt)

    return {
        "post": response.content,
        "iteration": state["iteration"] + 1
    }


# ============================================================
# REFLECT NODE
# ============================================================

def reflect_node(state: State):

    prompt = f"""
You are a professional LinkedIn content strategist.

Critically review the following LinkedIn post.

Original request:
{state["request"]}

LinkedIn post:
{state["post"]}

Evaluate:

1. Hook
2. Clarity
3. Professionalism
4. Engagement potential
5. Structure
6. Value to the reader
7. Call to action
8. Grammar
9. Whether the post satisfies the original request

Give specific and actionable improvements.

Do NOT rewrite the post.

Return only the critique.
"""

    response = llm.invoke(prompt)

    return {
        "critique": response.content
    }


# ============================================================
# ROUTER
# ============================================================

def should_continue(state: State):

    if state["iteration"] >= MAX_ITERATIONS:
        return "end"

    return "reflect"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(State)


# Add nodes

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_node(
    "reflect",
    reflect_node
)


# Entry point

workflow.set_entry_point(
    "generate"
)


# Generate → Reflect OR END

workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "reflect": "reflect",
        "end": END
    }
)


# Reflect → Generate

workflow.add_edge(
    "reflect",
    "generate"
)


# Compile graph

app = workflow.compile()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Reflection Agent",
    page_icon="✍️",
    layout="centered"
)

st.title("Reflection Agent")

st.write(
    "Generate → Reflect → Improve → Repeat"
)


request = st.text_area(
    "What LinkedIn post should the agent create?",
    placeholder=(
        "Write a LinkedIn post about getting "
        "a software developer internship."
    ),
    height=150
)


if st.button("Generate Post"):

    if not request.strip():

        st.warning(
            "Please enter a request."
        )

    else:

        initial_state = {
            "request": request,
            "post": "",
            "critique": "",
            "iteration": 0
        }

        with st.spinner(
            "Running Reflection Agent..."
        ):

            result = app.invoke(
                initial_state
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.subheader(
            "Final LinkedIn Post"
        )

        st.write(
            result["post"]
        )


        # ====================================================
        # ITERATION COUNT
        # ====================================================

        st.divider()

        st.write(
            f"Iterations completed: "
            f"{result['iteration']}"
        )


        # ====================================================
        # FINAL CRITIQUE
        # ====================================================

        with st.expander(
            "View Final Critique"
        ):

            st.write(
                result["critique"]
            )