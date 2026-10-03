import os
import glob
import io
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# =========================
# DATA CACHE
# =========================

DATASETS = {}


# =========================
# TOOLS
# =========================

@tool
def list_csv_files():
    """List all CSV files in the current project folder."""
    return [os.path.basename(f) for f in glob.glob("*.csv")]


@tool
def preload_datasets(files: list[str]):
    """Load CSV files into memory."""
    result = []
    for file in files:
        try:
            if file not in DATASETS:
                DATASETS[file] = pd.read_csv(file)
            result.append(f"{file}: {DATASETS[file].shape[0]} rows, {DATASETS[file].shape[1]} columns")
        except Exception as e:
            result.append(f"{file}: Error - {e}")
    return "\n".join(result)


@tool
def get_dataset_summary(file: str):
    """Return columns, shape and data types of a CSV dataset."""
    if file not in DATASETS:
        DATASETS[file] = pd.read_csv(file)

    df = DATASETS[file]
    return {
        "file": file,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict()
    }


@tool
def dataframe_analysis(file: str, operation: str):
    """Perform head, tail, describe or info operation on a dataset."""
    if file not in DATASETS:
        DATASETS[file] = pd.read_csv(file)

    df = DATASETS[file]

    if operation == "head":
        return df.head().to_string()

    if operation == "tail":
        return df.tail().to_string()

    if operation == "describe":
        return df.describe().to_string()

    if operation == "info":
        output = io.StringIO()
        df.info(buf=output)
        return output.getvalue()

    return "Allowed operations: head, tail, describe, info"


@tool
def classification(file: str, target: str):
    """Train a Random Forest classification model and return accuracy."""
    if file not in DATASETS:
        DATASETS[file] = pd.read_csv(file)

    df = DATASETS[file]

    if target not in df.columns:
        return f"Target column '{target}' does not exist."

    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True).fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    return {
        "task": "classification",
        "target": target,
        "accuracy": round(accuracy, 4)
    }


@tool
def regression(file: str, target: str):
    """Train a Random Forest regression model and return R2 and MSE."""
    if file not in DATASETS:
        DATASETS[file] = pd.read_csv(file)

    df = DATASETS[file]

    if target not in df.columns:
        return f"Target column '{target}' does not exist."

    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True).fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    return {
        "task": "regression",
        "target": target,
        "r2_score": round(r2_score(y_test, prediction), 4),
        "mean_squared_error": round(mean_squared_error(y_test, prediction), 4)
    }


# =========================
# LLM
# =========================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
    num_ctx=4096
)

tools = [
    list_csv_files,
    preload_datasets,
    get_dataset_summary,
    dataframe_analysis,
    classification,
    regression
]

llm_with_tools = llm.bind_tools(tools)

tool_map = {tool.name: tool for tool in tools}


# =========================
# STREAMLIT
# =========================

st.set_page_config(
    page_title="DataWizard",
    page_icon="📊"
)

st.title("📊 DataWizard")
st.write("Ask questions about your CSV datasets using local AI.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# =========================
# USER QUESTION
# =========================

question = st.chat_input("Ask something about your dataset...")


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    messages = [
        HumanMessage(content=question)
    ]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):

            for _ in range(5):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    answer = response.content
                    break

                for call in response.tool_calls:
                    tool = tool_map[call["name"]]

                    result = tool.invoke(call["args"])

                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=call["id"]
                        )
                    )

            else:
                answer = "Unable to complete the analysis."

            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })