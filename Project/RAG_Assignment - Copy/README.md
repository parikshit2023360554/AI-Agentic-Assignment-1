# RAG Assignment: Automated Policy Q&A System

## 1. Problem Statement
**Objective**: Build a Retrieval-Augmented Generation (RAG) system to answer employee questions based on internal company policies. Traditional search methods often fail to understand natural language queries or provide concise answers, leading to repetitive inquiries to HR. This system aims to provide accurate, context-aware answers by retrieving relevant policy sections and generating human-like responses.

## 2. Dataset / Knowledge Source
- **Type**: Plain Text (`.txt`).
- **Source**: Mock "Remote Work Policy" document (`knowledge_base.txt`). This sample contains rules on eligibility, work hours, equipment, and reimbursement.

## 3. RAG Architecture Block Diagram

```mermaid
graph LR
    UserQuery[User Query] --> EmbedQuery[Embedding Model]
    EmbedQuery --> VectorSearch[Vector Search (FAISS)]
    VectorSearch --> Retrieve[Retrieve Top K Chunks]
    Retrieve --> Augment[Combine Query + Context]
    Augment --> LLM[LLM Generation (OpenAI/Mock)]
    LLM --> Response[Final Answer]

    subgraph "Knowledge Processing"
    Doc[Document Source] --> Load[Text Loader]
    Load --> Chunk[Text Splitter]
    Chunk --> EmbedDoc[Embedding Model]
    EmbedDoc --> VectorStore[Vector Database (FAISS)]
    end
```

## 4. Text Chunking Strategy
- **Chunk Size**: `500` characters.
- **Chunk Overlap**: `50` characters (10%).
- **Reasoning**: Policies are typically structured in short, concise paragraphs. A smaller chunk size ensures we capture specific policy rules without retrieving irrelevant adjacent policies. The overlap maintains context across sentence boundaries.

## 5. Embedding Details
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Reasoning**: This is a lightweight, high-performance model suitable for semantic search tasks. It creates dense vector representations (384 dimensions) that capture semantic meaning efficiently even on CPU.

## 6. Vector Database
- **Tool**: `FAISS` (Facebook AI Similarity Search).
- **Reasoning**: Highly efficient for similarity search and clustering of dense vectors. It is easy to set up locally without external dependencies or API keys.

## 7. Future Improvements
1.  **Semantic Reranking**: Implement a cross-encoder reranker (e.g., BGE-Reranker) to refine the top-k results before passing to the LLM.
2.  **Hybrid Search**: Combine keyword-based search (BM25) with vector search to handle specific keyword queries better.
3.  **Metadata Filtering**: Tag chunks with metadata (e.g., "Policy Type", "Date") to allow users to filter by specific categories.
4.  **UI Integration**: Build a simple Streamlit or Chainlit interface for user interaction.

## 8. Requirements & Setup
### Tools Used
- `langchain`: Orchestration framework.
- `faiss-cpu`: Vector database.
- `sentence-transformers`: Embedding model.
- `openai` (Optional): For generation (can be swapped with local LLMs).

### Instructions to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook rag_assignment.ipynb
    ```
3.  Run all cells to see the pipeline in action.
