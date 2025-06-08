from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from data.qdrant_docs import documents
import ollama

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "test-base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

points = [
    models.PointStruct(
        id=i,
        vector=embedder.encode(doc).tolist(),
        payload={"text": doc}
    )
    for i, doc in enumerate(documents)
]
client.upsert(collection_name=COLLECTION_NAME, points=points)


def rag_pipeline(query, n_context=2, model="llama3"):
    query_vector = embedder.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=n_context
    )
    context = "\n".join(hit.payload['text'] for hit in results)
    prompt = f"Odpowiedz na pytanie korzystając z kontekstu:\n\n{context}\n\nPytanie: {query}\nOdpowiedź:"
    result = ollama.generate(model=model, prompt=prompt)
    return result['response']

