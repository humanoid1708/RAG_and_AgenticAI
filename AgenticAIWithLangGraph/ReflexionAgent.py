import os
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import streamlit as st

os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cpu")

# Configuration
MAX_ITERATIONS = 3
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# State
class State(TypedDict):
    question: str
    answer: str
    critique: str
    knowledge: str
    iteration: int
    continue_reflection: bool

# Nodes
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
        answer, remaining = content.split("CRITIQUE:", 1)
        answer = answer.replace("ANSWER:", "").strip()
        if "NEEDS_RESEARCH:" in remaining:
            critique, research = remaining.split("NEEDS_RESEARCH:", 1)
            critique = critique.strip()
            if "YES" in research.upper():
                needs_research = "YES"
        else:
            critique = remaining.strip()

    return {"answer": answer, "critique": critique, "continue_reflection": (needs_research == "YES")}

def retrieve_knowledge(state: State):
    knowledge_base = """
    LANGGRAPH
    LangGraph is a framework for building stateful AI workflows.
    State: State stores information shared between nodes.
    Nodes: Nodes are functions that perform individual tasks.
    Edges: Edges connect nodes and determine the workflow.
    Conditional Edges: Conditional edges allow the workflow to choose different paths based on the current state.
    Cycles: LangGraph supports cycles. A workflow can return to an earlier node and perform another iteration.

    REFLEXION
    Reflexion is an agent pattern in which an agent:
    1. Generates an initial response.
    2. Reflects on or critiques the response.
    3. Obtains additional information when needed.
    4. Revises the response.
    5. Repeats the process when necessary.

    The purpose of the loop is to improve the quality of the final response.
    """
    return {"knowledge": knowledge_base}

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
        answer, decision = content.split("SATISFACTORY:", 1)
        answer = answer.strip()
        if "YES" in decision.upper():
            satisfactory = True
    else:
        answer = content.strip()

    return {"answer": answer, "continue_reflection": not satisfactory}

def increment_iteration(state: State):
    return {"iteration": state["iteration"] + 1}

def should_continue(state: State):
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "continue" if state["continue_reflection"] else "end"

# Graph
workflow = StateGraph(State)
workflow.add_node("responder", responder)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("revisor", revisor)
workflow.add_node("increment", increment_iteration)

workflow.set_entry_point("responder")
workflow.add_edge("responder", "retrieve")
workflow.add_edge("retrieve", "revisor")
workflow.add_edge("revisor", "increment")
workflow.add_conditional_edges("increment", should_continue, {"continue": "responder", "end": END})

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Reflexion Agent", page_icon="🧠", layout="centered")
st.title("Reflexion Agent")
st.write("The agent generates an answer, critiques it, retrieves additional knowledge, revises it, and can repeat the process.")

question = st.text_input("Ask a question")

if st.button("Run Reflexion"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        initial_state = {
            "question": question,
            "answer": "",
            "critique": "",
            "knowledge": "",
            "iteration": 0,
            "continue_reflection": True,
        }
        with st.spinner("Running Reflexion Agent..."):
            result = app.invoke(initial_state)

        st.subheader("Final Answer")
        st.write(result["answer"])

        with st.expander("See Reflexion Details"):
            st.write("Iterations:", result["iteration"])
            st.write("Final critique:")
            st.write(result["critique"])
