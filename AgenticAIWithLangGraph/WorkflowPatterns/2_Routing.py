from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import streamlit as st

# LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")

# State
class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

# Nodes
def router_node(state: RouterState):
    prompt = f"""
You are an intent classifier.

Classify the user's request into exactly one category:

summarize
or
translate

Rules:
- If the user wants a summary, return: summarize
- If the user wants translation into French, return: translate

Return ONLY one word.

User request:
{state["user_input"]}
"""
    response = llm.invoke(prompt)
    task = response.content.strip().lower()
    task_type = "translate" if "translate" in task else "summarize"
    return {"task_type": task_type}

def summarize_node(state: RouterState):
    prompt = f"""
Summarize the following text clearly and concisely.

Text:
{state["user_input"]}

Return only the summary.
"""
    response = llm.invoke(prompt)
    return {"task_type": "summarize", "output": response.content}

def translate_node(state: RouterState):
    prompt = f"""
Translate the following text into French.

Text:
{state["user_input"]}

Return only the French translation.
"""
    response = llm.invoke(prompt)
    return {"task_type": "translate", "output": response.content}

def router(state: RouterState):
    return state["task_type"]

# Graph
workflow = StateGraph(RouterState)
workflow.add_node("router", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", router, {"summarize": "summarize", "translate": "translate"})
workflow.add_edge("summarize", END)
workflow.add_edge("translate", END)

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Routing Agent", page_icon="2", layout="centered")
st.title("Routing Agent")
st.write("Router → Specialized Node")

user_input = st.text_area("Enter your request", height=200, placeholder="Example: Summarize this article...")

if st.button("Run Router"):
    if not user_input.strip():
        st.warning("Please enter something.")
    else:
        with st.spinner("Routing request..."):
            result = app.invoke({"user_input": user_input, "task_type": "", "output": ""})

        st.subheader("Selected Route")
        st.code(result["task_type"])

        st.subheader("Output")
        st.write(result["output"])
