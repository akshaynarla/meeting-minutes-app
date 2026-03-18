# src/backend/backend_rag.py
from typing import List

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class DocumentRAG:
    """Helper module to embed transcript chunks and retrieve them via FAISS."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not RAG_AVAILABLE:
            raise ImportError("faiss-cpu or sentence-transformers is not installed. RAG features are unavailable.")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[str]):
        """Vectorize a list of overlapping/small chunks and build the FAISS index."""
        if not chunks:
            return
        self.chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        # Normalize for Inner Product -> Cosine Similarity
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve the top k chunks that match the query."""
        if not self.index or not self.chunks:
            return []
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Max fetch constraints
        fetch_k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_emb, fetch_k)
        
        results = []
        for i in indices[0]:
            if i >= 0:
                results.append(self.chunks[i])
        return results
