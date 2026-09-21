import numpy as np
import ollama

# 1. Example of documents
documents = [
    "Bugs in software need debugging.",
    "QA engineers find programming bugs.",
    "Bugs are common during summer.",
    "Spiders are studied by arachnologists."
]

# 2. Create embeddings
def get_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return np.array(response["embedding"])


# Create embeddings for all documents
document_embeddings = []

for document in documents:
    embedding = get_embedding(document)
    document_embeddings.append(embedding)

document_embeddings = np.array(document_embeddings)

print("Number of documents:", len(documents))
print("Embedding shape:", document_embeddings.shape)

# 3. Define metrices
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def dot_product(a, b):
    return np.dot(a, b)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# 4. Create query
query = "Who studies spiders?"

query_embedding = get_embedding(query)

# 5. Compare with every example document
results = []

for i, document_embedding in enumerate(document_embeddings):

    distance = euclidean_distance(
        query_embedding,
        document_embedding
    )

    dot = dot_product(
        query_embedding,
        document_embedding
    )

    cosine = cosine_similarity(
        query_embedding,
        document_embedding
    )

    results.append({
        "document": documents[i],
        "euclidean_distance": distance,
        "dot_product": dot,
        "cosine_similarity": cosine
    })


# 6. Display
print("\nSimilarity Results")
print("=" * 70)

for result in results:
    print("\nDocument:", result["document"])
    print("Euclidean Distance :", result["euclidean_distance"])
    print("Dot Product        :", result["dot_product"])
    print("Cosine Similarity   :", result["cosine_similarity"])



# 7. Cosine Search in Descending order
results.sort(
    key=lambda x: x["cosine_similarity"],
    reverse=True
)

print("\n\nTop Similar Documents")
print("=" * 70)

for rank, result in enumerate(results, start=1):

    print(
        f"{rank}. "
        f"{result['document']} "
        f"(score = {result['cosine_similarity']:.4f})"
    )