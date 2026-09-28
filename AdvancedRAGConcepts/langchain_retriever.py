from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOllama(model="llama3.1:8b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document(
        page_content="""
        The company smoking policy prohibits smoking inside all office buildings.
        Employees may smoke only in designated outdoor smoking areas.
        Smoking is not permitted near entrances, elevators, or emergency exits.
        """,
        metadata={"category": "smoking", "department": "HR"}
    ),

    Document(
        page_content="""
        Employees are entitled to 20 paid vacation days per year.
        Vacation requests should normally be submitted at least two weeks
        before the planned leave.
        """,
        metadata={"category": "leave", "department": "HR"}
    ),

    Document(
        page_content="""
        Employees working remotely must be available between 10 AM and 5 PM.
        They must attend scheduled team meetings and maintain regular
        communication with their manager.
        """,
        metadata={"category": "remote_work", "department": "HR"}
    ),

    Document(
        page_content="""
        Company laptops must be protected with a strong password.
        Employees must not install unauthorized software.
        Lost or stolen laptops must be reported to the IT department immediately.
        """,
        metadata={"category": "security", "department": "IT"}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policies"
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)

query = "Where can employees smoke?"

docs = retriever.invoke(query)

context = "\n\n".join(
    doc.page_content for doc in docs
)

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print(response.content)

