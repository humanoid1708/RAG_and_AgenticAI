from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser

llm = ChatOllama(model="llama3.1:8b")

prompt = PromptTemplate.from_template(
    "Explain in brief about {topic}."
)

prompt_json = PromptTemplate.from_template(
    "Explain about {library} python library specifying definition, most_used_functions, latest_version in a json format."
)

prompt_csv = PromptTemplate.from_template(
    "List only the top 5 subjects under the topic {subject}. Do not include any descriptions"
)

chain1 = prompt | llm | StrOutputParser()
chain2 = prompt_json | llm | JsonOutputParser()
chain3 = prompt_csv | llm | CommaSeparatedListOutputParser()

response1 = chain1.invoke({"topic": "RAG"})

'''
JsonOutputParser only checks:
Is the output valid JSON? ✅
Can it be converted into a Python object? ✅
It does not verify that required keys like "definition" or "latest_version" are present.
'''
response2 = chain2.invoke({"library": "PySpark"})


'''Part the sentence at every comma'''
response3 = chain3.invoke({"subject" : "Full Stack Development"})

print(response1)
print(response2)
print(response3)