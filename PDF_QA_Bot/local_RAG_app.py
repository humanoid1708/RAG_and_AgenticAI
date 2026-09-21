#To check how embeddings work
"""
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("What is machine learning?")
print(len(vector))
print(vector[:10])
"""

#Actual RAG code
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import gradio as gr

#PDF Loader
loader = PyPDFLoader("PDF_QA_Bot/document/sample.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

#LLM setup
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template("""
You are a helpful question-answering assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{input}

Answer:
""")


document_chain = create_stuff_documents_chain(
    llm,
    prompt
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

#Output
"""
response_1 = rag_chain.invoke({
    "input": "What is this color of sky?"
})

print(response_1["answer"], '\n')

response_2 = rag_chain.invoke({
    "input": "What is this document about?"
})

print(response_2["answer"])
"""

def ask_question(question):

    response = rag_chain.invoke({
        "input": question
    })

    return response["answer"]

interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        label="Ask a question"
    ),
    outputs=gr.Textbox(
        label="Answer"
    ),
    title="Local PDF QA Bot"
)

interface.launch()