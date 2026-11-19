from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import streamlit as st

# Configuration
MAX_ITERATIONS = 3
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")

# State
class State(TypedDict):
    request: str
    post: str
    critique: str
    iteration: int

# Nodes
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

If a previous post and critique exist, improve the previous post using the critique.

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
    return {"post": response.content, "iteration": state["iteration"] + 1}

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
    return {"critique": response.content}

def should_continue(state: State):
    return "end" if state["iteration"] >= MAX_ITERATIONS else "reflect"

# Graph
workflow = StateGraph(State)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("generate")
workflow.add_conditional_edges("generate", should_continue, {"reflect": "reflect", "end": END})
workflow.add_edge("reflect", "generate")

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Reflection Agent", page_icon="✍️", layout="centered")
st.title("Reflection Agent")
st.write("Generate → Reflect → Improve → Repeat")

request = st.text_area(
    "What LinkedIn post should the agent create?",
    placeholder="Write a LinkedIn post about getting a software developer internship.",
    height=150,
)

if st.button("Generate Post"):
    if not request.strip():
        st.warning("Please enter a request.")
    else:
        initial_state = {"request": request, "post": "", "critique": "", "iteration": 0}
        with st.spinner("Running Reflection Agent..."):
            result = app.invoke(initial_state)

        st.subheader("Final LinkedIn Post")
        st.write(result["post"])

        st.divider()
        st.write(f"Iterations completed: {result['iteration']}")

        with st.expander("View Final Critique"):
            st.write(result["critique"])
