import os

# Force Ollama to use CPU
os.environ["OLLAMA_LLM_LIBRARY"] = "cpu"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent
)


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Visualization Agent",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. PAGE TITLE
# ============================================================

st.title("📊 AI Data Visualization Agent")

st.write(
    "Upload a CSV file and ask questions about your data "
    "using natural language. The local Llama model can "
    "analyze the data and create visualizations."
)


# ============================================================
# 3. CSV UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ============================================================
# 4. STOP IF NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a CSV file to start."
    )

    st.stop()


# ============================================================
# 5. LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Could not read the CSV file:\n\n{e}"
    )

    st.stop()


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

st.subheader("Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


# ============================================================
# 7. SHOW DATA
# ============================================================

with st.expander("Preview Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


with st.expander("Column Information"):

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )


# ============================================================
# 8. LOCAL OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


# ============================================================
# 9. CREATE PANDAS DATAFRAME AGENT
# ============================================================

agent = create_pandas_dataframe_agent(

    llm=llm,

    df=df,

    verbose=False,

    # Allows the agent to execute Python code
    allow_dangerous_code=True,

    # Return the intermediate steps so that
    # we can inspect the generated Python code
    return_intermediate_steps=True,

    # Helps recover from parsing problems
    handle_parsing_errors=True,

    prefix="""
You are an expert data analyst.

You are working with a pandas DataFrame called df.

Your job is to answer questions about the dataset.

When the user asks for a visualization:

1. Analyze the requested columns.
2. Write Python code using pandas and matplotlib or seaborn.
3. Execute the code.
4. Create the requested chart.
5. Give a short explanation of the result.

Use the actual column names from the DataFrame.

Do not invent columns.

For visualizations, use matplotlib or seaborn.

Always make the chart readable with:
- meaningful title
- axis labels
- appropriate figure size

The user may request:
- bar charts
- line charts
- pie charts
- scatter plots
- box plots
- histograms
- heatmaps

You can also answer normal questions about the dataset.
"""
)


# ============================================================
# 10. SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# 11. DISPLAY PREVIOUS QUESTIONS
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# 12. USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask something about your dataset..."
)


# ============================================================
# 13. RUN AGENT
# ============================================================

if question:

    # -----------------------------
    # Display user question
    # -----------------------------

    st.session_state.chat_history.append({

        "role": "user",

        "content": question

    })

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------
    # Agent response
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your dataset..."
        ):

            try:

                # Clear any old matplotlib figures
                plt.close("all")


                # -----------------------------
                # Invoke agent
                # -----------------------------

                response = agent.invoke(
                    question
                )


                # -----------------------------
                # Get final answer
                # -----------------------------

                answer = response.get(
                    "output",
                    "No answer generated."
                )


                # -----------------------------
                # Display answer
                # -----------------------------

                st.markdown(answer)


                # ====================================================
                # DISPLAY GENERATED VISUALIZATION
                # ====================================================

                figures = plt.get_fignums()

                if figures:

                    st.subheader(
                        "Generated Visualization"
                    )

                    fig = plt.gcf()

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )


                # ====================================================
                # SHOW GENERATED PYTHON CODE
                # ====================================================

                intermediate_steps = response.get(
                    "intermediate_steps",
                    []
                )


                generated_code = []


                for step in intermediate_steps:

                    try:

                        action = step[0]

                        tool_input = action.tool_input


                        if isinstance(
                            tool_input,
                            dict
                        ):

                            code = tool_input.get(
                                "query",
                                ""
                            )

                        else:

                            code = str(
                                tool_input
                            )


                        if code:

                            generated_code.append(
                                code
                            )

                    except Exception:

                        pass


                if generated_code:

                    with st.expander(
                        "View Python code generated by Llama"
                    ):

                        for code in generated_code:

                            st.code(
                                code,
                                language="python"
                            )


                # -----------------------------
                # Save assistant message
                # -----------------------------

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": answer

                })


            except Exception as e:

                st.error(
                    f"""
                    Something went wrong.

                    Error:
                    {e}
                    """
                )