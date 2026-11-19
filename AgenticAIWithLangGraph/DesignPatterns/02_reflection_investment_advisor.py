from typing import Literal, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import streamlit as st

# LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.7, base_url="http://127.0.0.1:11434")

# Risk grades
Grades = Literal["ultra-conservative", "conservative", "moderate", "aggressive", "high risk"]

# Graph state
class State(TypedDict, total=False):
    investment_plan: str
    investor_profile: str
    target_grade: Grades
    feedback: str
    grade: Grades
    n: int

# Determine target grade
grade_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an investment risk-profile classifier.

Based on the investor profile, select exactly one target risk grade:

ultra-conservative
conservative
moderate
aggressive
high risk

Return ONLY the grade text.""",
    ),
    ("human", "Investor profile:\n{investor_profile}"),
])

grade_pipe = grade_prompt | llm

def determine_target_grade(state: State):
    response = grade_pipe.invoke({"investor_profile": state["investor_profile"]})
    grade = response.content.strip().lower()
    valid_grades = {"ultra-conservative", "conservative", "moderate", "aggressive", "high risk"}
    matched_grade = next((g for g in valid_grades if g in grade), "moderate")
    return {"target_grade": matched_grade}

# Initial generator
cathie_wood_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an educational investment-strategy generator
inspired by a bold, innovation-focused investing style.

Create a hypothetical investment strategy based on the investor profile.

Consider:
- long-term growth
- diversification
- emerging technologies
- risk tolerance
- investment horizon

Do not guarantee returns.
Clearly present this as an educational hypothetical strategy.""",
    ),
    ("human", "Investor profile:\n{investor_profile}"),
])

cathie_wood_pipe = cathie_wood_prompt | llm

# Evaluator schema
class Feedback(BaseModel):
    grade: Grades = Field(description="Risk classification of the investment plan.")
    feedback: str = Field(description="Concise reasoning for the assigned risk grade.")

# Refinement generator
ray_dalio_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an educational investment-strategy generator
focused on diversification, economic scenarios, and risk management.

Create a NEW hypothetical investment strategy after reviewing the
previous evaluator feedback.

Address the evaluator's concerns.

Possible adjustments:
- Too conservative → increase growth exposure appropriately.
- Too aggressive → add defensive assets and reduce speculative exposure.
- Poor diversification → diversify by asset class, geography, and sector.
- Inflation concern → consider inflation-sensitive assets.
- Excessive complexity → simplify the portfolio.

Do not guarantee returns.
This is educational content, not personalized financial advice.""",
    ),
    (
        "human",
        """Investor profile:
{investor_profile}

Previous strategy grade:
{grade}

Evaluator feedback:
{feedback}

Create a revised hypothetical strategy.""",
    ),
])

ray_dalio_pipe = ray_dalio_prompt | llm

# Generator node
def investment_plan_generator(state: State):
    if state.get("feedback"):
        response = ray_dalio_pipe.invoke({
            "investor_profile": state["investor_profile"],
            "grade": state["grade"],
            "feedback": state["feedback"],
        })
    else:
        response = cathie_wood_pipe.invoke({"investor_profile": state["investor_profile"]})
    return {"investment_plan": response.content}

# Evaluator
evaluator_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an investment risk evaluator.

Evaluate the hypothetical investment strategy using:

- capital preservation
- volatility
- diversification
- speculative exposure
- alignment with investor risk tolerance
- long-term suitability

Assign exactly one grade:

ultra-conservative
conservative
moderate
aggressive
high risk

Return structured feedback.""",
    ),
    (
        "human",
        """Investor profile:
{investor_profile}

Target risk grade:
{target_grade}

Investment plan:
{investment_plan}""",
    ),
])

buffett_evaluator_pipe = evaluator_prompt | llm.with_structured_output(Feedback)

def evaluate_plan(state: State):
    current_count = state.get("n", 0) + 1
    result = buffett_evaluator_pipe.invoke({
        "investment_plan": state["investment_plan"],
        "investor_profile": state["investor_profile"],
        "target_grade": state["target_grade"],
    })
    return {"grade": result.grade, "feedback": result.feedback, "n": current_count}

# Reflection routing
def route_investment(state: State, iteration_limit: int = 5):
    current_grade = state.get("grade")
    target_grade = state.get("target_grade")
    iterations = state.get("n", 0)

    if current_grade == target_grade or iterations >= iteration_limit:
        return "Accepted"
    return "Rejected + Feedback"

# Build LangGraph
builder = StateGraph(State)

builder.add_node("determine_target_grade", determine_target_grade)
builder.add_node("investment_plan_generator", investment_plan_generator)
builder.add_node("evaluate_plan", evaluate_plan)

builder.add_edge(START, "determine_target_grade")
builder.add_edge("determine_target_grade", "investment_plan_generator")
builder.add_edge("investment_plan_generator", "evaluate_plan")
builder.add_conditional_edges(
    "evaluate_plan",
    route_investment,
    {"Accepted": END, "Rejected + Feedback": "investment_plan_generator"},
)

workflow = builder.compile()

# Streamlit UI
st.set_page_config(page_title="Reflection Investment Advisor", page_icon="🔄", layout="wide")
st.title("Reflection Investment Advisor")
st.caption("LangGraph + Ollama | Generate → Evaluate → Refine → Repeat")

st.warning("Educational demonstration only. The generated investment plan is hypothetical and is not financial advice.")

st.subheader("Investor Profile")

profile = st.text_area(
    "Enter investor information",
    value=(
        "Age: 29\n"
        "Salary: $110,000\n"
        "Assets: $40,000\n"
        "Goal: Achieve financial independence by age 45\n"
        "Risk tolerance: High"
    ),
    height=180,
)

iteration_limit = st.slider("Maximum reflection iterations", min_value=1, max_value=5, value=5)

if st.button("Generate Investment Plan", type="primary"):
    if not profile.strip():
        st.warning("Please enter an investor profile.")
        st.stop()

    try:
        with st.spinner("Running generate → evaluate → refine loop..."):
            result = workflow.invoke(
                {
                    "investor_profile": profile,
                    "investment_plan": "",
                    "target_grade": "moderate",
                    "feedback": "",
                    "grade": "moderate",
                    "n": 0,
                },
                config={"recursion_limit": max(10, iteration_limit * 3 + 5)},
            )

        st.success("Reflection workflow completed.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Grade", result.get("target_grade", "N/A"))
        with col2:
            st.metric("Final Grade", result.get("grade", "N/A"))
        with col3:
            st.metric("Iterations", result.get("n", 0))

        st.subheader("Final Investment Plan")
        st.write(result.get("investment_plan", ""))

        st.subheader("Evaluator Feedback")
        st.write(result.get("feedback", ""))

        with st.expander("Investor Profile"):
            st.text(result.get("investor_profile", ""))

    except Exception as e:
        st.error("Could not run the workflow. Make sure Ollama is running and llama3.1:8b is available.")
        st.exception(e)

st.divider()
st.caption("Pattern: Reflection | Key LangGraph concepts: evaluator, conditional routing, feedback loop, iterative refinement")
