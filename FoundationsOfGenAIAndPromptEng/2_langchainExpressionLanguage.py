"""
2. LangChain Expression Language (LCEL)
---------------------------------------
Demonstrates how LangChain composes components as `Runnable` objects using LCEL:
  1. RunnableLambda (wrapping Python functions as Runnables)
  2. RunnableSequence vs. Pipe Operator (`|`)
  3. RunnableParallel (branching one input into multiple parallel steps)
  4. RunnablePassthrough & `.assign()` (forwarding input while adding derived fields)
  5. Combining PromptTemplate + LCEL + Batch Execution (`.invoke()` and `.batch()`)

Note: Uses pure Python Runnables so you can run and experiment with LCEL
directly without needing an LLM server running.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)


# 1. RunnableLambda: Turning Python Functions into Runnables
clean_text = RunnableLambda(lambda text: text.strip().lower())
split_words = RunnableLambda(lambda text: text.split())
count_words = RunnableLambda(lambda words: len(words))
print("=== 1. RunnableLambda (.invoke) ===")
sample = "   LangChain Expression Language makes chaining easy!   "
print("Input  :", repr(sample))
print("Cleaned:", repr(clean_text.invoke(sample)))



# 2. RunnableSequence vs. Pipe Operator (`|`)
explicit_sequence = RunnableSequence(clean_text, split_words, count_words)
pipe_sequence = clean_text | split_words | count_words
print("\n=== 2. RunnableSequence vs Pipe (`|`) ===")
print("RunnableSequence result:", explicit_sequence.invoke(sample))
print("Pipe (`|`) result      :", pipe_sequence.invoke(sample))

# 3. RunnableParallel: Running Multiple Branches on the Same Inpu
text_analysis_chain = clean_text | RunnableParallel(
    word_count=split_words | count_words,
    char_count=RunnableLambda(len),
    uppercase=RunnableLambda(str.upper),
)

print("\n=== 3. RunnableParallel ===")
analysis = text_analysis_chain.invoke(sample)
for key, value in analysis.items():
    print(f"  {key}: {value}")

# 4. RunnablePassthrough & `.assign()`
enrich_chain = RunnablePassthrough.assign(
    word_count=RunnableLambda(lambda d: len(d["text"].split())),
    preview=RunnableLambda(lambda d: d["text"][:20] + "..."),
)
print("\n=== 4. RunnablePassthrough.assign() ===")
enriched = enrich_chain.invoke({
    "topic": "LCEL",
    "text": "LCEL lets you compose prompts, models, and parsers cleanly.",
})
print(enriched)



# 5. Putting It Together: PromptTemplate + Pipe + `.batch()`
prompt = PromptTemplate.from_template("Summarize {topic} for a {audience}.")
prompt_to_str = RunnableLambda(lambda pv: pv.to_string())
mock_llm = RunnableLambda(lambda text: f"[Mock LLM Response] -> {text}")
mini_chain = prompt | prompt_to_str | mock_llm
print("\n=== 5. Composing PromptTemplate with LCEL (.invoke & .batch) ===")
single_result = mini_chain.invoke({"topic": "Prompt Engineering", "audience": "beginner"})
print("Single invoke:", single_result)

batch_inputs = [
    {"topic": "Embeddings", "audience": "data engineer"},
    {"topic": "AI Agents", "audience": "backend developer"},
]
batch_results = mini_chain.batch(batch_inputs)
print("Batch results:")
for item in batch_results:
    print(" ", item)
