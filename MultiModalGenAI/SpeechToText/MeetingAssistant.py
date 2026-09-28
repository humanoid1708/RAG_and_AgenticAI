import streamlit as st
import whisper
from langchain_ollama import ChatOllama

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️")
st.title("🎙️ AI Meeting Assistant")
st.write("Upload a meeting recording to generate a transcript, meeting minutes and tasks.")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
    num_ctx=4096
)
    return whisper_model, llm

whisper_model, llm = load_models()

audio = st.file_uploader("Upload meeting audio", type=["mp3", "wav", "m4a", "mp4"])

if audio:
    with open("meeting_audio", "wb") as f:
        f.write(audio.getbuffer())

    if st.button("Generate Meeting Notes"):
        with st.spinner("Transcribing audio..."):
            result = whisper_model.transcribe("meeting_audio", fp16=False)
            transcript = result["text"]

        st.subheader("Raw Transcript")
        st.write(transcript)

        with st.spinner("Analyzing meeting with Llama..."):
            prompt = f"""
You are an AI meeting assistant. Analyze this transcript and produce:
1. Cleaned Transcript
2. Meeting Overview
3. Key Points Discussed
4. Decisions Made
5. Important Dates/Deadlines
6. Task List with Task, Assignee and Deadline.

Do not invent information. If an assignee or deadline is not mentioned, write "Not specified".

Transcript:
{transcript}
"""
            response = llm.invoke(prompt)

        st.subheader("Meeting Minutes & Task List")
        st.write(response.content)