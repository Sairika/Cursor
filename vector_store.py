"""
Vector store module for the RAG Chatbot system.
Handles document embeddings and similarity search using FAISS and ChromaDB.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    # Create dummy classes for fallback
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts):
            return np.random.rand(len(texts), 384)
    
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

class VectorStore:
    """Handles document embeddings and similarity search."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_type: str = "faiss",
                 db_path: str = "./vector_db"):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model
            vector_db_type: Type of vector database ("faiss" or "chromadb")
            db_path: Path to store the vector database
        """
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db_type
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model {embedding_model}: {e}")
            print("Using fallback random embeddings")
            self.embedder = None
        
        # Initialize vector database
        self._init_vector_db()
    
    def _init_vector_db(self):
        """Initialize the vector database based on type."""
        if self.vector_db_type == "faiss":
            self._init_faiss_db()
        elif self.vector_db_type == "chromadb":
            self._init_chromadb()
        else:
            raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")
    
    def _init_faiss_db(self):
        """Initialize FAISS database."""
        self.faiss_index = None
        self.documents = []
        self.metadata = []
        
        # Try to load existing index
        index_path = self.db_path / "faiss_index.bin"
        docs_path = self.db_path / "documents.pkl"
        meta_path = self.db_path / "metadata.pkl"
        
        if index_path.exists() and docs_path.exists() and meta_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(index_path))
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded existing FAISS index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Warning: Could not load existing FAISS index: {e}")
                self._create_new_faiss_index()
        else:
            self._create_new_faiss_index()
    
    def _create_new_faiss_index(self):
        """Create a new FAISS index."""
        if self.embedder is None:
            # Use random vectors for fallback
            dimension = 384
        else:
            dimension = self.embedder.get_sentence_embedding_dimension()
        
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        print(f"Created new FAISS index with dimension {dimension}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB database."""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path / "chromadb"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Try to get existing collection
            try:
                self.chroma_collection = self.chroma_client.get_collection("documents")
                print("Loaded existing ChromaDB collection")
            except:
                # Create new collection
                self.chroma_collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"description": "Document embeddings for RAG chatbot"}
                )
                print("Created new ChromaDB collection")
                
        except Exception as e:
            print(f"Warning: Could not initialize ChromaDB: {e}")
            print("Falling back to FAISS")
            self.vector_db_type = "faiss"
            self._init_faiss_db()
    
    def add_documents(self, documents: List[Any]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            return False
        
        try:
            if self.vector_db_type == "faiss":
                return self._add_documents_faiss(documents)
            elif self.vector_db_type == "chromadb":
                return self._add_documents_chromadb(documents)
            else:
                return False
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def _add_documents_faiss(self, documents: List[Any]) -> bool:
        """Add documents to FAISS database."""
        try:
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            if self.embedder is not None:
                embeddings = self.embedder.encode(texts)
            else:
                # Fallback: random embeddings
                embeddings = np.random.rand(len(texts), 384)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents.extend(documents)
            self.metadata.extend([doc.metadata for doc in documents])
            
            print(f"Added {len(documents)} documents to FAISS index")
            return True
            
        except Exception as e:
            print(f"Error adding documents to FAISS: {e}")
            return False
    
    def _add_documents_chromadb(self, documents: List[Any]) -> bool:
        """Add documents to ChromaDB."""
        try:
            # Extract text content and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Generate embeddings
            if self.embedder is not None:
                embeddings = self.embedder.encode(texts).tolist()
            else:
                # Fallback: random embeddings
                embeddings = np.random.rand(len(texts), 384).tolist()
            
            # Add to ChromaDB
            self.chroma_collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            if self.vector_db_type == "faiss":
                return self._search_faiss(query, top_k)
            elif self.vector_db_type == "chromadb":
                return self._search_chromadb(query, top_k)
            else:
                return []
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _search_faiss(self, query: str, top_k: int) -> List[Tuple[Any, float]]:
        """Search in FAISS database."""
        if self.faiss_index is None or len(self.documents) == 0:
            return []
        
        try:
            # Generate query embedding
            if self.embedder is not None:
                query_embedding = self.embedder.encode([query])
            else:
                # Fallback: random embedding
                query_embedding = np.random.rand(1, 384)
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            return results
            
        except Exception as e:
            print(f"Error searching FAISS: {e}")
            return []
    
    def _search_chromadb(self, query: str, top_k: int) -> List[Tuple[Any, float]]:
        """Search in ChromaDB."""
        try:
            # Generate query embedding
            if self.embedder is not None:
                query_embedding = self.embedder.encode([query]).tolist()
            else:
                # Fallback: random embedding
                query_embedding = np.random.rand(1, 384).tolist()
            
            # Search
            results = self.chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['distances']:
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Create document object
                    from document_processor import Document
                    document = Document(
                        page_content=doc,
                        metadata=results['metadatas'][0][results['documents'][0].index(doc)]
                    )
                    
                    formatted_results.append((document, similarity))
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []
    
    def save(self) -> bool:
        """Save the vector store to disk."""
        try:
            if self.vector_db_type == "faiss":
                return self._save_faiss()
            elif self.vector_db_type == "chromadb":
                return self._save_chromadb()
            else:
                return False
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False
    
    def _save_faiss(self) -> bool:
        """Save FAISS index to disk."""
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.db_path / "faiss_index.bin"))
                
                with open(self.db_path / "documents.pkl", 'wb') as f:
                    pickle.dump(self.documents, f)
                
                with open(self.db_path / "metadata.pkl", 'wb') as f:
                    pickle.dump(self.metadata, f)
                
                print("FAISS index saved successfully")
                return True
            return False
            
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False
    
    def _save_chromadb(self) -> bool:
        """ChromaDB saves automatically, so just return True."""
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            if self.vector_db_type == "faiss":
                return {
                    "type": "FAISS",
                    "total_documents": len(self.documents),
                    "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
                    "embedding_dimension": self.faiss_index.d if self.faiss_index else 0
                }
            elif self.vector_db_type == "chromadb":
                return {
                    "type": "ChromaDB",
                    "total_documents": self.chroma_collection.count(),
                    "collection_name": self.chroma_collection.name
                }
            else:
                return {"type": "Unknown", "error": "No database initialized"}
                
        except Exception as e:
            return {"type": "Error", "error": str(e)}
    
    def clear(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            if self.vector_db_type == "faiss":
                self._create_new_faiss_index()
                self.documents = []
                self.metadata = []
            elif self.vector_db_type == "chromadb":
                self.chroma_client.delete_collection("documents")
                self.chroma_collection = self.chroma_client.create_collection("documents")
            
            print("Vector store cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            return False