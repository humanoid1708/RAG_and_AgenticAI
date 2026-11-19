"""
1. Prompt Templates in LangChain
--------------------------------
Demonstrates how to build reusable, structured prompts using `langchain_core.prompts`:
  1. Basic PromptTemplate (`.format()` and `.invoke()`)
  2. Partial Prompt Templates (pre-filling fixed variables)
  3. ChatPromptTemplate (system, human, and AI message roles)
  4. MessagesPlaceholder (injecting dynamic message lists like chat history)
  5. FewShotPromptTemplate (guiding outputs with input/output examples)
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

# 1. Basic PromptTemplate
prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

# `.format()` returns a plain Python string.
formatted_str = prompt_template.format(adjective="funny", content="chickens")
print("=== 1. Basic PromptTemplate (.format) ===")
print(formatted_str)

# `.invoke()` treats the template as a LangChain Runnable and returns a PromptValue.
prompt_value = prompt_template.invoke({"adjective": "short", "content": "programmers"})
print("\n=== 1b. Basic PromptTemplate (.invoke) ===")
print(repr(prompt_value))
print("As string:", prompt_value.to_string())



# 2. Partial Prompt Templates (Pre-filling Variables)
base_template = PromptTemplate.from_template(
    "Explain {topic} to a {audience} in {language}."
)
beginner_english_template = base_template.partial(
    audience="beginner",
    language="simple English",
)

print("\n=== 2. Partial PromptTemplate ===")
print(beginner_english_template.format(topic="Retrieval-Augmented Generation (RAG)"))



# 3. ChatPromptTemplate (Role-Based Messages)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role} who answers concisely."),
    ("human", "What is {concept}?"),
])

chat_messages = chat_template.format_messages(
    role="AI tutor",
    concept="prompt engineering",
)

print("\n=== 3. ChatPromptTemplate ===")
for msg in chat_messages:
    print(f"[{msg.type.upper()}] {msg.content}")


# 4. MessagesPlaceholder (Dynamic Message Lists / Chat History)
conversation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly coding assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chat_history = [
    HumanMessage(content="Hi! I am learning LangChain."),
    AIMessage(content="Welcome! What topic would you like to start with?"),
]

conversation_messages = conversation_template.format_messages(
    history=chat_history,
    question="How do prompt templates help?",
)

print("\n=== 4. ChatPromptTemplate with MessagesPlaceholder ===")
for msg in conversation_messages:
    print(f"[{msg.type.upper()}] {msg.content}")


# 5. FewShotPromptTemplate (Prompting by Example)
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "fast", "antonym": "slow"},
]

example_prompt = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input word.",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
)

print("\n=== 5. FewShotPromptTemplate ===")
print(few_shot_template.format(input="bright"))
