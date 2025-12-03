import os
import torch
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialise ChromaDB client
        # self.client = chromadb.PersistentClient(path="./chroma_db")
        self.client = chromadb.Client()

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, title: str = '', chunk_size: int = 1000):
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # For this, we will use LangChain's RecursiveCharacterTextSplitter
        # because it automatically handles sentence boundaries and preserves context better
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,  # ~200 words per chunk
                chunk_overlap=200,  # Overlap to preserve context
                separators=["\n\n", "\n", " ", "", ". "],
            )

            chunks = text_splitter.split_text(text)

            # Add metadata to each chunk
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append(
                    {
                        "content": chunk,               
                        "title": title,
                        "chunk_index": f"{title}_{i}",     
                    }
                )
        except Exception as e:
            print("Error during text chunking:", e)

        return chunk_data

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of dicts with keys {"content": str, "metadata": dict}
        """

        print(f"Processing {len(documents)} documents...")

        # Load embedding model once
        try:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

            embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

        except Exception as e:
            print("Error loading embedding model:", e)

        # Start ID
        next_id = self.collection.count()

        # Process each document
        for doc_index, doc in enumerate(documents):
            content = doc.page_content

            title = doc.metadata.get("source", "")

            chunked_document = self.chunk_text(content)

            text_chunks = [chunk["content"] for chunk in chunked_document]

            # Create IDs
            ids = [f"doc_{next_id}_chunk_{i}" for i in range(len(text_chunks))]

            # Create embeddings for all chunks
            try:
                embeddings = embedding_model.encode(text_chunks).tolist()

            except Exception as e:
                print(f"Error generating embeddings:", e)

            # Push to Chroma
            try:
                print(f"Adding document to vector DB...")
                self.collection.add(
                    ids=ids,
                    documents=text_chunks,
                    embeddings=embeddings,
                    metadatas=chunked_document,
                )
            except Exception as e:
                print(f"Error adding document {doc_index} to vector DB:", e)
            next_id += len(text_chunks)
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.
        """

        print(f"Retrieving relevant documents for query: {query}")

        relevant_results = {
            "ids": [],
            "documents": [],
            "distances": [],
            "metadatas": [],
        }

        print("Embedding query...")
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        print("Querying collection...")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        if not results or not results.get("ids"):
            print("No results found.")
            return relevant_results

        print("Filtering results...")
        for i, distance in enumerate(results["distances"][0]):
            if distance < 0.4:
                relevant_results["ids"].append(results["ids"][0][i])
                relevant_results["documents"].append(results["documents"][0][i])
                relevant_results["distances"].append(distance)
                relevant_results["metadatas"].append(results["metadatas"][0][i])

        return relevant_results
