import streamlit as st

from youtube_utils import get_transcript
from rag import create_vectorstore, ask_question


st.set_page_config(
    page_title="YouTube RAG",
    page_icon="▶️"
)

st.title("YouTube Summarizer & Q&A")

url = st.text_input(
    "YouTube URL",
    placeholder="Paste a YouTube URL here"
)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if st.button("Load Video"):

    if not url:
        st.error("Please enter a YouTube URL.")

    else:

        with st.spinner("Fetching transcript..."):

            try:

                transcript = get_transcript(url)

                st.session_state.vectorstore = (
                    create_vectorstore(transcript)
                )

                st.success(
                    "Video loaded successfully."
                )

            except Exception as e:

                st.error(str(e))


question = st.text_input(
    "Ask a question about the video"
)


if st.button("Ask"):

    if st.session_state.vectorstore is None:

        st.warning("Load a video first.")

    elif not question:

        st.warning("Enter a question.")

    else:

        with st.spinner("Thinking..."):

            answer = ask_question(
                st.session_state.vectorstore,
                question
            )

        st.write("### Answer")
        st.write(answer)