from pathlib import Path

from . import config
from .ingest import load_folder, load_pdf
from .llm import generate
from .vectorstore import VectorStore

NO_DOCS_MESSAGE = "No documents are indexed yet. Please upload a PDF first"
NOT_FOUND_MESSAGE = "I couldn't find that in the support documents. Try rephrasing your question."
class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()

    def ingest_pdf(self, source, name=None):
        return self.store.add_chunks(load_pdf(source, name))

    def ingest_folder(self, folder=config.DATA_DIR):
        return self.store.add_chunks(load_folder(folder))

    def list_sources(self):
        return self.store.list_sources()

    def reset(self):
        self.store.reset()

    @staticmethod
    def _retrieval_query(question, history):
        if history and len(question.split()) <= 6:
            previous = [h["content"] for h in history if h["role"] == "user"]
            if previous:
                return previous[-1] + " " + question
        return question

    def ask(self, question, history=None):
        if self.store.count() == 0:
            return {"answer": NO_DOCS_MESSAGE, "sources": []}

        chunks = self.store.search(self._retrieval_query(question, history), config.TOP_K)
        if not chunks or chunks[0]["score"] < config.MIN_SCORE:
            return {"answer": NOT_FOUND_MESSAGE, "sources": []}

        try:
            answer = generate(question, chunks, history)
        except Exception as exc:
            answer = f"⚠️ The language model could not answer: {exc}"
        return {"answer": answer,"sources":chunks}
    