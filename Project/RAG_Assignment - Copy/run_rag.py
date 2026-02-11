import os
from typing import List
from langchain_community.document_loaders import TextLoader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import textwrap

def main():
    print("--- RAG Project Execution ---\n")

    # 1. Check Data Source
    file_path = 'knowledge_base.txt'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    # 2. Load Documents
    print(f"Loading '{file_path}'...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    
    # 3. Chunking Strategy
    print("\nApplying Chunking Strategy...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    print(f"Example chunk preview:\n{textwrap.indent(chunks[0].page_content, '  ')}\n")

    # 4. Embeddings (Local)
    print("Initializing Embedding Model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 5. Vector Store
    print("Creating FAISS Vector Store...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    print("Vector Store created successfully.")

    # 6. Mock LLM for Demonstration
    class MockLLM:
        def __init__(self, vector_store):
            self.vector_store = vector_store

        def answer_question(self, query):
            # Retrieve
            docs = self.vector_store.similarity_search(query, k=2)
            
            # Simulate Generation
            response = f"""
            [Generated Answer based on Context]
            The policy states that... (simulated extraction)
            
            Source Context 1: {docs[0].page_content[:100]}...
            Source Context 2: {docs[1].page_content[:100]}...
            (This is a mock response demonstrating the retrieval pipeline.)
            """
            return response, docs

    rag_system = MockLLM(vector_store)

    # 7. Test Queries
    test_queries = [
        "What is the eligibility for remote work?",
        "Does the company pay for internet?",
        "What is the policy on public Wi-Fi?"
    ]

    print("\n--- Running Test Queries ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer, source_docs = rag_system.answer_question(query)
        print(f"Answer:{textwrap.indent(answer, '  ')}")

if __name__ == "__main__":
    main()
