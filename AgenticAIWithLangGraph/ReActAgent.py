from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
import streamlit as st

# LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")

# Tools
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression. Example: 25 * 48"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Could not calculate the expression."

@tool
def weather(city: str) -> str:
    """Get weather information for a city."""
    weather_data = {
        "mumbai": "Mumbai: 30°C, humid, partly cloudy.",
        "delhi": "Delhi: 27°C, dry and sunny.",
        "pune": "Pune: 25°C, pleasant and partly cloudy.",
        "bangalore": "Bangalore: 23°C, cloudy with a chance of rain.",
        "london": "London: 12°C, cloudy with light rain.",
        "zurich": "Zurich: 8°C, cloudy and cold.",
    }
    city = city.lower().strip()
    return weather_data.get(city, f"No weather data available for {city}.")

tools = [calculator, weather]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Nodes
def agent_node(state: AgentState):
    system_message = """
You are a helpful ReAct AI assistant.

You follow the ReAct pattern:
Reason → Act → Observe → Reason → Answer

Your job is to:
1. Understand the user's question.
2. Decide whether a tool is required.
3. Call the appropriate tool when necessary.
4. Examine the tool result.
5. Call another tool if necessary.
6. Give the final answer when enough information is available.

Available tools:
1. calculator - Use this for mathematical calculations.
2. weather - Use this for weather information.

Important:
- Use tools when they are appropriate.
- Do not invent tool results.
- If a tool provides information, use that information.
- If no tool is required, answer directly.
"""
    messages = [HumanMessage(content=system_message)]
    messages.extend(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    outputs = []
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name not in tools_by_name:
            outputs.append(ToolMessage(content=f"Unknown tool: {tool_name}", name=tool_name, tool_call_id=tool_call["id"]))
            continue

        result = tools_by_name[tool_name].invoke(tool_args)
        outputs.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"]))
    return {"messages": outputs}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="ReAct AI Agent", page_icon="🤖", layout="centered")
st.title("ReAct AI Agent")
st.write("Reason → Act → Observe → Reason → Answer")

question = st.text_input("Ask something", placeholder="Example: What is 125 * 37?")

if st.button("Run Agent"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        initial_state = {"messages": [HumanMessage(content=question)]}
        with st.spinner("Agent is working..."):
            result = app.invoke(initial_state)

        messages = result["messages"]
        st.subheader("Final Answer")
        st.write(messages[-1].content)

        with st.expander("View Agent Steps"):
            for i, message in enumerate(messages):
                st.markdown(f"### Step {i + 1}")
                if isinstance(message, HumanMessage):
                    st.write("User:")
                    st.write(message.content)
                elif isinstance(message, AIMessage):
                    if message.tool_calls:
                        st.write("Agent decided to use a tool:")
                        for call in message.tool_calls:
                            st.code(f"Tool: {call['name']}\nArguments: {call['args']}")
                    else:
                        st.write("Agent:")
                        st.write(message.content)
                elif isinstance(message, ToolMessage):
                    st.write("Tool result:")
                    st.write(message.content)
