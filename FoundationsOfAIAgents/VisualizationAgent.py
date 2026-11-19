import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Force Ollama to use CPU to prevent CUDA crashes on Windows
os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cpu")

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 1. Streamlit Configuration & Header
st.set_page_config(page_title="AI Visualization Agent", page_icon="📊", layout="wide")
st.title("📊 AI Data Visualization Agent")
st.write(
    "Analyze CSV datasets and generate charts using natural language with a local Llama model."
)


# 2. Dataset Selection (Bundled CSV or Upload)
col_mode, col_sel = st.columns([1, 2])
with col_mode:
    source_choice = st.radio(
        "Dataset Source:",
        ["Bundled Dataset", "Upload CSV"],
        horizontal=True,
    )

df = None
dataset_name = ""

if source_choice == "Bundled Dataset":
    bundled_files = ["classification-dataset.csv", "regression-dataset.csv"]
    with col_sel:
        selected_file = st.selectbox("Select bundled dataset:", bundled_files)
    dataset_name = selected_file
    file_path = os.path.join(BASE_DIR, selected_file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        st.error(f"File not found: {file_path}")
        st.stop()
else:
    with col_sel:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to start.")
        st.stop()
    dataset_name = uploaded_file.name
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file:\n\n{e}")
        st.stop()


# 3. Dataset Overview
st.subheader(f"Dataset: {dataset_name}")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

with st.expander("Preview Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Column Information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
    })
    st.dataframe(info_df, use_container_width=True)


# 4. Agent Setup
llm = ChatOllama(model="llama3.1:8b", temperature=0)

agent_prefix = """
You are an expert data analyst working with a pandas DataFrame called df.
Your job is to answer questions about the dataset accurately.

When the user asks for a visualization:
1. Identify the relevant columns from df (use actual column names, do not invent them).
2. Write clean Python code using pandas, matplotlib, or seaborn.
3. Make charts readable with a meaningful title, axis labels, and appropriate figure size.
4. Always execute the code to generate the figure.
5. Provide a short explanation of what the chart shows.

Available chart types include bar charts, line charts, scatter plots, histograms, box plots, heatmaps, and pie charts.
You can also answer direct statistical or summary questions about the dataset.
"""

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=False,
    allow_dangerous_code=True,
    return_intermediate_steps=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    prefix=agent_prefix,
)


# 5. Chat History & User Interaction
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your dataset (e.g., 'Plot the distribution of prices' or 'Show a bar chart of education vs target')...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your dataset..."):
            try:
                plt.close("all")
                response = agent.invoke(question)
                answer = response.get("output", "No answer generated.")
                st.markdown(answer)

                # Display any generated matplotlib/seaborn figures
                if plt.get_fignums():
                    st.subheader("Generated Visualization")
                    st.pyplot(plt.gcf(), use_container_width=True)

                # Show generated Python code steps
                steps = response.get("intermediate_steps", [])
                generated_code = []
                for step in steps:
                    try:
                        action = step[0]
                        tool_input = action.tool_input
                        code = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                        if code:
                            generated_code.append(code)
                    except Exception:
                        pass

                if generated_code:
                    with st.expander("View Python code generated by Llama"):
                        for code in generated_code:
                            st.code(code, language="python")

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Something went wrong.\n\nError:\n{e}")
