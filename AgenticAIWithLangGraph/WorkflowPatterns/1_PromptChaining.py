from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import streamlit as st

# LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")

# State
class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

# Nodes
def generate_resume_summary(state: ChainState):
    prompt = f"""
You are a professional resume assistant.

Read the following job description.

Identify the most important:
- Technical skills
- Required experience
- Responsibilities
- Qualifications
- Important keywords

Then create a strong resume summary for a candidate who wants to apply for this role.

Job Description:
{state["job_description"]}

Return only the resume summary.
"""
    response = llm.invoke(prompt)
    return {"resume_summary": response.content}

def generate_cover_letter(state: ChainState):
    prompt = f"""
You are a professional cover letter writer.

Write a personalized cover letter using:

JOB DESCRIPTION:
{state["job_description"]}

RESUME SUMMARY:
{state["resume_summary"]}

Requirements:
- Professional
- Concise
- Personalized
- Clearly connect the candidate's skills with the job requirements
- Do not invent experience
- Do not mention AI

Return only the cover letter.
"""
    response = llm.invoke(prompt)
    return {"cover_letter": response.content}

# Graph
workflow = StateGraph(ChainState)
workflow.add_node("resume_summary", generate_resume_summary)
workflow.add_node("cover_letter", generate_cover_letter)

workflow.set_entry_point("resume_summary")
workflow.add_edge("resume_summary", "cover_letter")
workflow.add_edge("cover_letter", END)

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Prompt Chaining", page_icon="1", layout="centered")
st.title("Prompt Chaining")
st.write("Job Description → Resume Summary → Cover Letter")

job_description = st.text_area("Paste Job Description", height=250)

if st.button("Generate Application"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        initial_state = {"job_description": job_description, "resume_summary": "", "cover_letter": ""}
        with st.spinner("Running workflow..."):
            result = app.invoke(initial_state)

        st.subheader("Resume Summary")
        st.write(result["resume_summary"])

        st.divider()
        st.subheader("Cover Letter")
        st.write(result["cover_letter"])
