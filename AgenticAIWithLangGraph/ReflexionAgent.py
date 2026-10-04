import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, END

import os

os.environ["OLLAMA_LLM_LIBRARY"] = "cpu"
from langchain_ollama import ChatOllama

# ============================================================
# CONFIGURATION
# ============================================================

MAX_ITERATIONS = 3

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    question: str
    answer: str
    critique: str
    knowledge: str
    iteration: int
    continue_reflection: bool


# ============================================================
# NODE 1: RESPONDER
# ============================================================

def responder(state: State):

    prompt = f"""
You are an AI research assistant.

Answer the user's question as accurately as possible.

Question:
{state["question"]}

Previous answer:
{state["answer"]}

Previous critique:
{state["critique"]}

Additional knowledge:
{state["knowledge"]}

Your job:

1. Produce the best answer you can.
2. Critically examine your answer.
3. Identify missing information.
4. Identify information that may be incorrect or unsupported.
5. Decide whether additional research would improve the answer.

Return exactly:

ANSWER:
<answer>

CRITIQUE:
<critique>

NEEDS_RESEARCH:
YES or NO
"""

    response = llm.invoke(prompt)

    content = response.content

    answer = content
    critique = ""
    needs_research = "NO"

    if "CRITIQUE:" in content:

        answer, remaining = content.split(
            "CRITIQUE:",
            1
        )

        answer = answer.replace(
            "ANSWER:",
            ""
        ).strip()

        if "NEEDS_RESEARCH:" in remaining:

            critique, research = remaining.split(
                "NEEDS_RESEARCH:",
                1
            )

            critique = critique.strip()

            if "YES" in research.upper():

                needs_research = "YES"

        else:

            critique = remaining.strip()

    return {
        "answer": answer,
        "critique": critique,
        "continue_reflection": (
            needs_research == "YES"
        )
    }


# ============================================================
# NODE 2: RETRIEVE KNOWLEDGE
# ============================================================

def retrieve_knowledge(state: State):

    # Local knowledge base.
    # This replaces Tavily for our beginner version.

    knowledge_base = """
    LANGGRAPH

    LangGraph is a framework for building stateful
    AI workflows.

    State:
    State stores information shared between nodes.

    Nodes:
    Nodes are functions that perform individual tasks.

    Edges:
    Edges connect nodes and determine the workflow.

    Conditional Edges:
    Conditional edges allow the workflow to choose
    different paths based on the current state.

    Cycles:
    LangGraph supports cycles. A workflow can return
    to an earlier node and perform another iteration.

    REFLEXION

    Reflexion is an agent pattern in which an agent:

    1. Generates an initial response.
    2. Reflects on or critiques the response.
    3. Obtains additional information when needed.
    4. Revises the response.
    5. Repeats the process when necessary.

    The purpose of the loop is to improve the quality
    of the final response.
    """

    return {
        "knowledge": knowledge_base
    }


# ============================================================
# NODE 3: REVISOR
# ============================================================

def revisor(state: State):

    prompt = f"""
You are the Revisor in a Reflexion Agent.

User question:
{state["question"]}

Current answer:
{state["answer"]}

Self-critique:
{state["critique"]}

Additional knowledge:
{state["knowledge"]}

Improve the current answer.

Requirements:

1. Correct weaknesses identified in the critique.
2. Use the additional knowledge when relevant.
3. Remove unsupported claims.
4. Make the answer clear and accurate.
5. Do not mention the internal Reflexion process.
6. Return only the improved answer.

At the end, on a separate line, write:

SATISFACTORY: YES

or

SATISFACTORY: NO
"""

    response = llm.invoke(prompt)

    content = response.content

    satisfactory = False

    if "SATISFACTORY:" in content:

        answer, decision = content.split(
            "SATISFACTORY:",
            1
        )

        answer = answer.strip()

        if "YES" in decision.upper():

            satisfactory = True

    else:

        answer = content.strip()

    return {
        "answer": answer,
        "continue_reflection": not satisfactory
    }


# ============================================================
# NODE 4: INCREMENT ITERATION
# ============================================================

def increment_iteration(state: State):

    return {
        "iteration": state["iteration"] + 1
    }


# ============================================================
# ROUTER
# ============================================================

def should_continue(state: State):

    # Safety condition:
    # never allow more than MAX_ITERATIONS.

    if state["iteration"] >= MAX_ITERATIONS:

        return "end"

    if state["continue_reflection"]:

        return "continue"

    return "end"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(State)


# Add nodes

workflow.add_node(
    "responder",
    responder
)

workflow.add_node(
    "retrieve",
    retrieve_knowledge
)

workflow.add_node(
    "revisor",
    revisor
)

workflow.add_node(
    "increment",
    increment_iteration
)


# Entry point

workflow.set_entry_point(
    "responder"
)


# First pass

workflow.add_edge(
    "responder",
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "revisor"
)

workflow.add_edge(
    "revisor",
    "increment"
)


# Reflexion loop

workflow.add_conditional_edges(

    "increment",

    should_continue,

    {
        "continue": "responder",
        "end": END
    }
)


# Compile

app = workflow.compile()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Reflexion Agent",
    page_icon="🧠",
    layout="centered"
)

st.title("Reflexion Agent")

st.write(
    "The agent generates an answer, critiques it, "
    "retrieves additional knowledge, revises it, "
    "and can repeat the process."
)


question = st.text_input(
    "Ask a question"
)


if st.button("Run Reflexion"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        initial_state = {

            "question": question,

            "answer": "",

            "critique": "",

            "knowledge": "",

            "iteration": 0,

            "continue_reflection": True
        }


        with st.spinner(
            "Running Reflexion Agent..."
        ):

            result = app.invoke(
                initial_state
            )


        # ==============================================
        # FINAL ANSWER
        # ==============================================

        st.subheader(
            "Final Answer"
        )

        st.write(
            result["answer"]
        )


        # ==============================================
        # DEBUG INFORMATION
        # ==============================================

        with st.expander(
            "See Reflexion Details"
        ):

            st.write(
                "Iterations:",
                result["iteration"]
            )

            st.write(
                "Final critique:"
            )

            st.write(
                result["critique"]
            )