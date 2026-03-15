# Enterprise RAG Framework 🧠

A high-performance, modular Retrieval-Augmented Generation (RAG) framework designed for enterprise-scale knowledge search. Optimized for low-latency retrieval and scalable ingestion across heterogeneous data sources.

## Features

- **Scalable Ingestion**: Asynchronous processing of PDF, Markdown, and API-based data sources.
- **Advanced Retrieval**: Hybrid search (Vector + Keyword) with re-ranking (Cross-Encoders).
- **Multi-Cloud Support**: Integrated with AWS Bedrock (Llama 3.1) and Azure OpenAI.
- **Observability**: Built-in tracing with Langfuse for retrieval quality and cost monitoring.
- **Sub-Second Latency**: Optimized vector store indices (Qdrant/Pinecone) for millisecond retrieval.

## Architecture

`	ext
[ Data Source ] -> [ Ingestion Pipeline ] -> [ Vector DB (Qdrant) ]
                                                   |
[ User Query ] -> [ Hybrid Retriever ] -> [ Reranker ] -> [ LLM (Bedrock) ]
`

## Core Modules

- src/ingestion/: Parallelized document parsing and embedding generation.
- src/retrieval/: Query expansion, semantic search, and re-ranking logic.
- src/api/: FastAPI-based orchestration layer with session management.

## Performance

- **Retrieval Latency**: <150ms for 1M+ document chunks.
- **Generation Latency**: <800ms end-to-end using streaming.

---
[LinkedIn](https://linkedin.com/in/navneet-beri) | [Main Profile](https://github.com/NavneetBeridev)