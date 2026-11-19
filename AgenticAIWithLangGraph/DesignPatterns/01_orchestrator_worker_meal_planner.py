import operator
from typing import Annotated, List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
import streamlit as st

# LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.7, base_url="http://127.0.0.1:11434")

# Structured schemas
class Dish(BaseModel):
    name: str = Field(description="Name of the dish.")
    ingredients: List[str] = Field(description="Ingredients needed for the dish.")
    location: str = Field(description="Cuisine or cultural origin of the dish.")

class Dishes(BaseModel):
    sections: List[Dish] = Field(description="One structured section for every requested meal.")

# Orchestrator prompt
dish_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the head chef and kitchen orchestrator.

The user will provide a list of meals. Break the request into one
structured dish for every meal.

For every dish provide:
- dish name
- ingredients
- cuisine or cultural origin

Do not merge different meals into one dish.""",
    ),
    ("human", "Meals requested:\n{meals}"),
])

planner_pipe = dish_prompt | llm.with_structured_output(Dishes)

# Main graph state
class State(TypedDict, total=False):
    meals: str
    sections: List[Dish]
    completed_menu: Annotated[List[str], operator.add]
    final_meal_guide: str

# Orchestrator node
def orchestrator(state: State):
    result = planner_pipe.invoke({"meals": state["meals"]})
    return {"sections": result.sections}

# Worker state
class WorkerState(TypedDict):
    section: Dish
    completed_menu: Annotated[List[str], operator.add]

# Worker prompt
chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a world-class chef from {location}.

Prepare a detailed cooking guide for the dish below.

Include:
1. A short chef introduction.
2. Ingredients.
3. Preparation steps.
4. Cooking steps.
5. Practical cooking tips.

Dish: {name}

Ingredients:
{ingredients}""",
    ),
])

chef_pipe = chef_prompt | llm

# Dynamic fan-out using Send()
def assign_workers(state: State):
    """Create one worker task for every dish."""
    return [Send("chef_worker", {"section": section}) for section in state["sections"]]

# Chef worker
def chef_worker(state: WorkerState):
    section = state["section"]
    response = chef_pipe.invoke({
        "name": section.name,
        "location": section.location,
        "ingredients": ", ".join(section.ingredients),
    })
    return {"completed_menu": [response.content]}

# Synthesizer
def synthesizer(state: State):
    completed = "\n\n---\n\n".join(state.get("completed_menu", []))
    return {"final_meal_guide": completed}

# Build LangGraph
builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("chef_worker", chef_worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["chef_worker"])
builder.add_edge("chef_worker", "synthesizer")
builder.add_edge("synthesizer", END)

workflow = builder.compile()

# Streamlit UI
st.set_page_config(page_title="Orchestrator-Worker Meal Planner", page_icon="🍳", layout="wide")
st.title("Orchestrator–Worker Meal Planner")
st.caption("LangGraph + Ollama | Orchestrator → Parallel Workers → Synthesizer")

st.markdown("""
Enter multiple meals. The **orchestrator** breaks them into separate
tasks, `Send()` creates one chef worker per task, and the workers
produce their cooking guides before the synthesizer combines them.
""")

meals = st.text_area(
    "Meals",
    value="Steak and eggs, tacos, and chili",
    height=100,
    placeholder="Example: biryani, pasta, tacos",
)

if st.button("Generate Meal Guide", type="primary"):
    if not meals.strip():
        st.warning("Please enter at least one meal.")
        st.stop()

    try:
        with st.spinner("Orchestrating meal tasks..."):
            result = workflow.invoke({"meals": meals, "sections": [], "completed_menu": [], "final_meal_guide": ""})

        st.success("Meal guide generated.")
        st.subheader("Orchestrator Output")

        for i, section in enumerate(result.get("sections", []), start=1):
            with st.expander(f"{i}. {section.name} — {section.location}"):
                st.write("**Ingredients**")
                for ingredient in section.ingredients:
                    st.write(f"- {ingredient}")

        st.subheader("Final Meal Guide")
        st.markdown(result["final_meal_guide"])

    except Exception as e:
        st.error("Could not run the workflow. Make sure Ollama is running and llama3.1:8b is available.")
        st.exception(e)

st.divider()
st.caption("Pattern: Orchestrator–Worker | Key LangGraph concept: Send()")
