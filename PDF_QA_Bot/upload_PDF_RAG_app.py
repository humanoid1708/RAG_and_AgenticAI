from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import gradio as gr

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

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

def ask_question(file, question):

    loader = PyPDFLoader(file)

    documents = loader.load()

    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    response = rag_chain.invoke({
        "input": question
    })

    return response["answer"]

interface = gr.Interface(
    fn=ask_question,

    inputs=[
        gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="filepath"
        ),

        gr.Textbox(
            label="Ask a question"
        )
    ],

    outputs=gr.Textbox(
        label="Answer"
    ),

    title="Local PDF QA Bot"
)

interface.launch()