import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

from . import config

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self._get_collection()
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def _get_collection(self):
        return self.client.get_or_create_collection\
            (name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    def count(self):
        return self.collection.count()
    def add_chunks(self,chunks):
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
        ids = [
            hashlib.md5(f"{c['source']}|{c['page']}|{i}|{c['text']}".encode("utf-8")).hexdigest()
            for i, c in enumerate(chunks)
        ]
        metadatas = [{"source":c["source"], "page":c["page"]} for c in chunks]

        batch = 500
        for i in range(0, len(chunks), batch):
            self.collection.upsert(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                embeddings=embeddings[i : i+batch],
                metadatas=metadatas[i : i + batch],
            )
            return len(chunks)
    def search(self, query,k=config.TOP_K):
        total = self.count()
        if total == 0:
            return []
        if "bge" in config.EMBEDDING_MODEL.lower():
            query = _BGE_QUERY_PREFIX + query
        q_emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=min(k, total))
        results = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append(
                {
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"],
                    "score": max(0.0,1.0 - dist),
                }
            )
        return results

    def list_sources(self):
        if self.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"]})

    def reset(self):
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self._get_collection()