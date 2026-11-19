from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
import streamlit as st

# LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")

# State
class State(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str

# Nodes
def translate_french(state: State):
    prompt = f"Translate this text into French.\n\nText:\n{state['text']}\n\nReturn only the translation."
    response = llm.invoke(prompt)
    return {"french": response.content.strip()}

def translate_spanish(state: State):
    prompt = f"Translate this text into Spanish.\n\nText:\n{state['text']}\n\nReturn only the translation."
    response = llm.invoke(prompt)
    return {"spanish": response.content.strip()}

def translate_japanese(state: State):
    prompt = f"Translate this text into Japanese.\n\nText:\n{state['text']}\n\nReturn only the translation."
    response = llm.invoke(prompt)
    return {"japanese": response.content.strip()}

def aggregator(state: State):
    combined = f"""Original Text:
{state["text"]}

French:
{state["french"]}

Spanish:
{state["spanish"]}

Japanese:
{state["japanese"]}"""
    return {"combined_output": combined}

# Graph
workflow = StateGraph(State)
workflow.add_node("french", translate_french)
workflow.add_node("spanish", translate_spanish)
workflow.add_node("japanese", translate_japanese)
workflow.add_node("aggregator", aggregator)

workflow.add_edge(START, "french")
workflow.add_edge(START, "spanish")
workflow.add_edge(START, "japanese")

workflow.add_edge("french", "aggregator")
workflow.add_edge("spanish", "aggregator")
workflow.add_edge("japanese", "aggregator")
workflow.add_edge("aggregator", END)

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Parallelization", page_icon="3", layout="centered")
st.title("Parallelization Agent")
st.write("One input → Multiple independent tasks → Aggregator")

text = st.text_area("Enter English text", height=150, placeholder="Good morning! I hope you have a wonderful day.")

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running parallel translations..."):
            result = app.invoke({"text": text, "french": "", "spanish": "", "japanese": "", "combined_output": ""})

        st.subheader("Translations")
        st.write(result["combined_output"])
