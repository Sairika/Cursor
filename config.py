"""
Configuration file for the RAG Chatbot system.
Contains all the settings and parameters for document processing, vector storage, and LLM configuration.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the RAG Chatbot system."""
    
    # Document Processing Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHUNKS = 1000
    
    # Vector Database Settings
    VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "chromadb"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SIMILARITY_TOP_K = 5
    
    # LLM Settings
    DEFAULT_MODEL = "microsoft/DialoGPT-medium"  # Lightweight model for Colab
    ALTERNATIVE_MODELS = {
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "falcon-7b": "tiiuae/falcon-7b-instruct",
        "mpt-7b": "mosaicml/mpt-7b-instruct"
    }
    
    # Model Parameters
    MODEL_PARAMS = {
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": 50256
    }
    
    # Memory Settings
    MEMORY_WINDOW = 10
    MAX_HISTORY_LENGTH = 1000
    
    # Prompt Templates
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided document context. 
    Always provide accurate information based on the context given. If you don't know the answer, say "I don't know" 
    rather than making up information. Be concise but helpful."""
    
    QA_PROMPT_TEMPLATE = """Context: {context}

Question: {question}

Answer based on the context above:"""
    
    SUMMARIZATION_PROMPT = """Please provide a comprehensive summary of the following document sections:

{context}

Summary:"""
    
    # File Paths
    VECTOR_DB_PATH = "./vector_db"
    CACHE_DIR = "./cache"
    SAMPLE_DOCS_DIR = "./sample_documents"
    
    # Supported File Extensions
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.md': 'markdown',
        '.html': 'html'
    }
    
    @classmethod
    def get_model_config(cls, model_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name is None:
            model_name = cls.DEFAULT_MODEL
            
        config = cls.MODEL_PARAMS.copy()
        
        # Model-specific configurations
        if "llama" in model_name.lower():
            config.update({
                "max_length": 2048,
                "temperature": 0.6
            })
        elif "mistral" in model_name.lower():
            config.update({
                "max_length": 4096,
                "temperature": 0.7
            })
        elif "falcon" in model_name.lower():
            config.update({
                "max_length": 1024,
                "temperature": 0.8
            })
            
        return config
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [cls.VECTOR_DB_PATH, cls.CACHE_DIR, cls.SAMPLE_DOCS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get configuration for embedding models."""
        return {
            "model_name": cls.EMBEDDING_MODEL,
            "cache_folder": cls.CACHE_DIR,
            "device": "cpu"  # Use CPU for Colab compatibility
        }