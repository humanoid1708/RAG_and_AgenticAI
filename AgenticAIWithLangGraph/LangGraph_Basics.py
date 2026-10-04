import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


# =============================
# LLM
# =============================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


# =============================
# STATE
# =============================

class State(TypedDict):
    question: str
    context: str
    answer: str
    relevant: bool


# =============================
# NODES
# =============================

def validate_question(state: State):

    if not state["question"].strip():

        return {
            "answer": "Please enter a question."
        }

    return {}


def provide_context(state: State):

    question = state["question"].lower()

    if "langgraph" in question:

        return {
            "context": """
            LangGraph is a Python framework for building
            stateful AI workflows.

            Its core concepts include:

            - State
            - Nodes
            - Edges
            - Conditional edges

            Nodes perform tasks.
            State carries information.
            Edges connect nodes.
            Conditional edges allow decisions.
            """,
            "relevant": True
        }

    return {
        "context": "",
        "relevant": False
    }


def router(state: State):

    if state["relevant"]:
        return "relevant"

    return "irrelevant"


def answer_question(state: State):

    prompt = f"""
You are a LangGraph tutor.

Answer the question using ONLY the context below.

Context:
{state["context"]}

Question:
{state["question"]}

Explain the answer simply for a beginner.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def reject_question(state: State):

    return {
        "answer":
        "I don't have enough context to answer that. "
        "Please ask something about LangGraph."
    }


# =============================
# GRAPH
# =============================

workflow = StateGraph(State)

workflow.add_node("validate", validate_question)
workflow.add_node("context", provide_context)
workflow.add_node("answer", answer_question)
workflow.add_node("reject", reject_question)

workflow.set_entry_point("validate")

workflow.add_edge("validate", "context")

workflow.add_conditional_edges(
    "context",
    router,
    {
        "relevant": "answer",
        "irrelevant": "reject"
    }
)

workflow.add_edge("answer", END)
workflow.add_edge("reject", END)

app = workflow.compile()


# =============================
# STREAMLIT
# =============================

st.set_page_config(
    page_title="LangGraph Tutor",
    page_icon="🧠"
)

st.title("LangGraph 101")
st.write("Ask a question about LangGraph.")

question = st.text_input(
    "Your question"
)

if st.button("Ask"):

    if question:

        result = app.invoke({
            "question": question,
            "context": "",
            "answer": "",
            "relevant": False
        })

        st.subheader("Answer")
        st.write(result["answer"])

    else:

        st.warning("Please enter a question.")