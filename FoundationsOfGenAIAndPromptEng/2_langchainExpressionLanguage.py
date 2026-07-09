from langchain_core.runnables import RunnableSequence, RunnableParallel

# chain = RunnableSequence([runnable1, runnable2]) 

# chain = RunnableParallel({
#     "key1" : runnable1,
#     "key2" : runnable2,
# })

# chain = runnable1 | runnable2