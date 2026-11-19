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

# Router
def router_node(state: RouterState):
    prompt = f"""
You are a request classification system.

Classify the user's request into EXACTLY ONE of these categories:
ride_hailing_call
restaurant_order
groceries
default_handler

Definitions:
ride_hailing_call: Requests involving taxis, cabs, rides, pickup, drop-off or transportation.
restaurant_order: Requests involving ordering food, restaurants, pizza, burgers, meals, etc.
groceries: Requests involving grocery shopping, milk, vegetables, bread, eggs, fruits, etc.
default_handler: Anything that does not belong to the three categories above.

User request:
{state["user_input"]}

Return ONLY the category name.
"""
    response = llm.invoke(prompt)
    result = response.content.strip().lower()

    if "ride_hailing_call" in result:
        task_type = "ride_hailing_call"
    elif "restaurant_order" in result:
        task_type = "restaurant_order"
    elif "groceries" in result:
        task_type = "groceries"
    else:
        task_type = "default_handler"

    return {"task_type": task_type}

# Specialized agent nodes
def ride_hailing_node(state: RouterState):
    prompt = f"""
You are a ride-hailing assistant.

Analyze the user's request and organize:
- Pickup location
- Destination
- Ride type
- Time
- Special requirements

User request:
{state["user_input"]}

Provide a clear ride request summary.
Do not invent missing information.
"""
    response = llm.invoke(prompt)
    return {"task_type": "ride_hailing_call", "output": response.content}

def restaurant_order_node(state: RouterState):
    prompt = f"""
You are a restaurant ordering assistant.

Analyze the user's request and organize:
- Food items
- Quantities
- Customizations
- Delivery or pickup
- Timing

User request:
{state["user_input"]}

Provide a clear restaurant order summary.
Do not invent missing information.
"""
    response = llm.invoke(prompt)
    return {"task_type": "restaurant_order", "output": response.content}

def groceries_node(state: RouterState):
    prompt = f"""
You are a grocery shopping assistant.

Analyze the user's request and organize:
- Grocery items
- Quantities
- Brand preferences
- Store preference
- Delivery information
- Special instructions

User request:
{state["user_input"]}

Provide a clear grocery order summary.
Do not invent missing information.
"""
    response = llm.invoke(prompt)
    return {"task_type": "groceries", "output": response.content}

def default_handler_node(state: RouterState):
    prompt = f"""
The user's request does not belong to:
- Ride hailing
- Restaurant ordering
- Grocery shopping

User request:
{state["user_input"]}

Politely explain that this system currently handles those three services.
"""
    response = llm.invoke(prompt)
    return {"task_type": "default_handler", "output": response.content}

def router(state: RouterState):
    return state["task_type"]

# Graph
workflow = StateGraph(RouterState)
workflow.add_node("router", router_node)
workflow.add_node("ride_hailing_call", ride_hailing_node)
workflow.add_node("restaurant_order", restaurant_order_node)
workflow.add_node("groceries", groceries_node)
workflow.add_node("default_handler", default_handler_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    router,
    {
        "ride_hailing_call": "ride_hailing_call",
        "restaurant_order": "restaurant_order",
        "groceries": "groceries",
        "default_handler": "default_handler",
    },
)

workflow.add_edge("ride_hailing_call", END)
workflow.add_edge("restaurant_order", END)
workflow.add_edge("groceries", END)
workflow.add_edge("default_handler", END)

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Multi-Agent Router", page_icon="4", layout="centered")
st.title("Multi-Agent Routing System")
st.write("Router → Specialized Agent")

user_input = st.text_area("What do you need?", height=150, placeholder="Example: I need a ride from Pune to the airport at 6 PM.")

if st.button("Process Request"):
    if not user_input.strip():
        st.warning("Please enter a request.")
    else:
        with st.spinner("Routing request..."):
            result = app.invoke({"user_input": user_input, "task_type": "", "output": ""})

        st.subheader("Selected Agent")
        st.code(result["task_type"])

        st.subheader("Agent Response")
        st.write(result["output"])
