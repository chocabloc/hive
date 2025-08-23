import chromadb
from chromadb.utils import embedding_functions as embed_fn

# some constants. should try other models maybe
DATA_PATH = "./data/"
EMBED_MODEL = "all-MiniLM-L6-v2"

# create the client and the fragments collection if it doesn't exist
# fragment id's are increases sequentially
client = chromadb.PersistentClient(path=DATA_PATH+"chroma/")
embed_func = embed_fn.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
fragments = client.get_or_create_collection(
    name="fragments",
    embedding_function=embed_func,
    metadata={"hnsw:space": "cosine"}
)
lastid = fragments.count()

# simple search and add functions

def search_for(text: str, topk: int):
    results = fragments.query(
        query_texts=text,
        n_results=topk
    )
    return [
        {
            "text": results['documents'][0][i],
            "distance": results['distances'][0][i],
            "metadata": results['metadatas'][0][i]
        }
        for i in range(0, topk)
    ]

def add(text: str, refers_to: str):
    global lastid
    fragments.add(
        documents=text,
        ids=f"fragment_{lastid}",
        metadatas={"refers_to": refers_to}
    )
    lastid += 1