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
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
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

        return chunk_data

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        We convert our text chunks into vector embeddings that capture semantic meaning
        so that similar texts are close in vector space.
        Each chunk becomes a 384-dimensional vector
        """
        device = (
            "cuda" 
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": device},
        )

        # model = HuggingFaceEmbeddings(
        #     model_name=self.embedding_model_name,
        #     model_kwargs={"device": "cpu"},  # pas de torch → device=cpu
        # )

        embeddings = model.embed_documents(documents)
        return embeddings

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # Now we store our chunks and their embeddings in ChromaDB for fast retrieval
        next_id = self.collection.count()

        for document in documents:
            chunked_document = self.chunk_text(document)    # to split each document into chunks
            embeddings = self.embed_documents(chunked_document)    # to get embeddings for each chunk
            ids = list(range(next_id, next_id + len(chunked_document)))
            ids = [f"doc_{id}" for id in ids]

            # We store the embeddings, documents, metadata, and IDs in the vector database
            # We're storing each chunk with its embedding and a unique ID.
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                documents=chunked_document,
            )
            next_id += len(chunked_document)
            # HINT: Print progress messages to inform the user
            # We're storing each chunk with its embedding and a unique ID.
            print(f"Processing {len(documents)} documents...")
            print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Convert question to vector
        print(f"Retrieving relevant documents for query: {query}")

        relevant_results = {
            "ids": [],
            "documents": [],
            "distances": [],
            "metadatas": [],
        }

        # Embed the query using the same model used for documents
        print("Embedding query...")
        query_embedding = self.embed_documents([query])[0]  # Get the first (and only) embedding

        print("Querying collection...")
        # Query the collection
        # 'query' will allow us to get the n_results nearest neighbor embeddings for provided query_embeddings or query_texts.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        print("Filtering results...")
        keep_item = [False] * len(results["ids"][0])
        for i, distance in enumerate(results["distances"][0]):
            if distance < 0.4:  # Example threshold for similarity
                keep_item[i] = True

        for i, keep in enumerate(keep_item):
            if keep:
                relevant_results["ids"].append(results["ids"][0][i])
                relevant_results["documents"].append(results["documents"][0][i])
                relevant_results["distances"].append(results["distances"][0][i])
                relevant_results["metadatas"].append(results["metadatas"][0][i])

        return relevant_results
