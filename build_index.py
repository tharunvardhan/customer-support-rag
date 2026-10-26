from src.rag import RAGPipeline
if __name__ == "__main__":
    rag = RAGPipeline()
    count = rag.ingest_folder()
    print(f"Indexed {count} chunks from: {', '.join(rag.list_sources()) or 'no files found in data/'}")


    