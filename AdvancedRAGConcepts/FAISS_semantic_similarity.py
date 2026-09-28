from langchain_ollama import OllamaEmbeddings
import faiss
import numpy as np

#Sample Document (generated)
documents = [
    "Python is a popular programming language used for software development and data science.",

    "Machine learning allows computers to learn patterns from data and make predictions.",

    "FAISS is a library developed for efficient similarity search over vectors.",

    "Football is a popular sport played between two teams of eleven players.",

    "Photography is the process of creating images using light and a camera.",

    "Deep learning uses neural networks with multiple layers to learn complex patterns."
]

#Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

document_embeddings = embeddings.embed_documents(documents)

document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)

print("Embedding shape:", document_embeddings.shape)

#FAISS index
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

print("Vectors stored in FAISS:", index.ntotal)


#Search function
def search(query, k=3):

    # Convert query to embedding
    query_embedding = embeddings.embed_query(query)

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        k
    )

    # Display results
    for rank, idx in enumerate(indices[0]):

        print(f"\nRank {rank + 1}")
        print("Distance:", distances[0][rank])
        print("Document:", documents[idx])


query = "How do computers learn from information?"

search(query)