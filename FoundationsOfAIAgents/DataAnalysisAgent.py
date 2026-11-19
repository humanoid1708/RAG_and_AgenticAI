import glob
import io
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Ensure CPU mode on Windows to avoid CUDA initialization issues
os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS: dict[str, pd.DataFrame] = {}


def _resolve_file_path(file: str) -> str:
    """Resolve file path whether executed from folder or repository root."""
    if os.path.exists(file):
        return file
    local_path = os.path.join(BASE_DIR, file)
    if os.path.exists(local_path):
        return local_path
    return file


def _get_df(file: str) -> pd.DataFrame:
    """Load and cache a dataframe from file."""
    if file not in DATASETS:
        resolved = _resolve_file_path(file)
        DATASETS[file] = pd.read_csv(resolved)
    return DATASETS[file]


# Tools
@tool
def list_csv_files() -> list[str]:
    """List all CSV files in the project folder."""
    files = set()
    for pattern in ("*.csv", os.path.join(BASE_DIR, "*.csv")):
        for f in glob.glob(pattern):
            files.add(os.path.basename(f))
    return sorted(files)


@tool
def preload_datasets(files: list[str]) -> str:
    """Load specified CSV files into memory cache."""
    result = []
    for file in files:
        try:
            df = _get_df(file)
            result.append(f"{file}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            result.append(f"{file}: Error - {e}")
    return "\n".join(result)


@tool
def get_dataset_summary(file: str) -> dict:
    """Return rows, columns, column names, and data types of a CSV dataset."""
    df = _get_df(file)
    return {
        "file": file,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }


@tool
def dataframe_analysis(file: str, operation: str) -> str:
    """Perform head, tail, describe, or info operation on a dataset."""
    df = _get_df(file)
    op = operation.strip().lower()

    if op == "head":
        return df.head().to_string()
    if op == "tail":
        return df.tail().to_string()
    if op == "describe":
        return df.describe().to_string()
    if op == "info":
        buf = io.StringIO()
        df.info(buf=buf)
        return buf.getvalue()

    return "Allowed operations: head, tail, describe, info"


@tool
def classification(file: str, target: str = "target") -> dict | str:
    """Train a Random Forest classification model and return accuracy.
    Default target is 'target' for classification-dataset.csv.
    """
    df = _get_df(file)
    if target not in df.columns:
        return f"Target column '{target}' does not exist in {file}. Available columns: {list(df.columns)}"

    df_clean = df.dropna(subset=[target])
    X = pd.get_dummies(df_clean.drop(columns=[target]), drop_first=True).fillna(0)
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return {
        "task": "classification",
        "file": file,
        "target": target,
        "accuracy": round(float(accuracy), 4),
    }


@tool
def regression(file: str, target: str = "price") -> dict | str:
    """Train a Random Forest regression model and return R2 score and MSE.
    Default target is 'price' for regression-dataset.csv.
    """
    df = _get_df(file)
    if target not in df.columns:
        return f"Target column '{target}' does not exist in {file}. Available columns: {list(df.columns)}"

    df_clean = df.dropna(subset=[target])
    X = pd.get_dummies(df_clean.drop(columns=[target]), drop_first=True).fillna(0)
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    return {
        "task": "regression",
        "file": file,
        "target": target,
        "r2_score": round(float(r2_score(y_test, pred)), 4),
        "mean_squared_error": round(float(mean_squared_error(y_test, pred)), 4),
    }


# LLM & Agent Setup
llm = ChatOllama(model="llama3.1:8b", temperature=0.1, num_ctx=4096)
tools = [
    list_csv_files,
    preload_datasets,
    get_dataset_summary,
    dataframe_analysis,
    classification,
    regression,
]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

SYSTEM_INSTRUCTION = (
    "You are DataWizard, an expert AI data analysis assistant. "
    "You have access to tools to inspect, summarize, analyze, and train machine learning models "
    "(Random Forest classification and regression) on CSV datasets. "
    "Always invoke the appropriate tools to answer user questions with real data. "
    "If you need to know which datasets are available, call list_csv_files first."
)


# Streamlit UI
st.set_page_config(page_title="DataWizard", page_icon="📊", layout="centered")
st.title("📊 DataWizard")
st.write("Ask questions about your CSV datasets and train ML models using a local AI agent.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

question = st.chat_input("Ask something about your dataset (e.g., 'What datasets are available?' or 'Run classification on classification-dataset.csv')...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=question),
    ]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer = "Unable to complete the analysis."
            for _ in range(5):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    answer = response.content
                    break

                for call in response.tool_calls:
                    tool_fn = tool_map.get(call["name"])
                    if not tool_fn:
                        messages.append(
                            ToolMessage(
                                content=f"Tool '{call['name']}' not found.",
                                tool_call_id=call["id"],
                            )
                        )
                        continue

                    try:
                        result = tool_fn.invoke(call["args"])
                    except Exception as err:
                        result = f"Error executing tool '{call['name']}': {err}"

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
