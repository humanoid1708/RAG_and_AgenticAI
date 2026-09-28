from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#Models
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

#FAISS Database - Used for chunking
def create_vectorstore(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.create_documents([transcript])

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

#Define prompt for answering questions
def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a helpful assistant answering questions
about a YouTube video.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available in the video.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content