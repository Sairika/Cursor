"""
Document processing module for the RAG Chatbot system.
Handles loading, parsing, and chunking of various document formats.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from langchain.document_loaders import (
        TextLoader, PDFMinerLoader, Docx2txtLoader, 
        UnstructuredMarkdownLoader, BSHTMLLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    # Fallback for environments where langchain is not available
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

class DocumentProcessor:
    """Handles document loading, parsing, and chunking operations."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                return self._load_text_file(file_path)
            elif file_extension == '.pdf':
                return self._load_pdf_file(file_path)
            elif file_extension == '.docx':
                return self._load_docx_file(file_path)
            elif file_extension == '.md':
                return self._load_markdown_file(file_path)
            elif file_extension == '.html':
                return self._load_html_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def _load_text_file(self, file_path: Path) -> List[Document]:
        """Load a text file."""
        try:
            loader = TextLoader(str(file_path))
            return loader.load()
        except Exception:
            # Fallback: read file manually
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": str(file_path)})]
    
    def _load_pdf_file(self, file_path: Path) -> List[Document]:
        """Load a PDF file."""
        try:
            loader = PDFMinerLoader(str(file_path))
            return loader.load()
        except Exception:
            # Fallback: try to extract text using PyPDF2 if available
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                return [Document(page_content=content, metadata={"source": str(file_path)})]
            except ImportError:
                raise Exception("PDF processing requires PyPDF2. Install with: pip install PyPDF2")
    
    def _load_docx_file(self, file_path: Path) -> List[Document]:
        """Load a Word document."""
        try:
            loader = Docx2txtLoader(str(file_path))
            return loader.load()
        except Exception:
            # Fallback: try to extract text using python-docx if available
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(str(file_path))
                content = ""
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
                return [Document(page_content=content, metadata={"source": str(file_path)})]
            except ImportError:
                raise Exception("DOCX processing requires python-docx. Install with: pip install python-docx")
    
    def _load_markdown_file(self, file_path: Path) -> List[Document]:
        """Load a Markdown file."""
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            return loader.load()
        except Exception:
            # Fallback: read file manually
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": str(file_path)})]
    
    def _load_html_file(self, file_path: Path) -> List[Document]:
        """Load an HTML file."""
        try:
            loader = BSHTMLLoader(str(file_path))
            return loader.load()
        except Exception:
            # Fallback: read file manually and strip HTML tags
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Simple HTML tag removal
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            return [Document(page_content=clean_content, metadata={"source": str(file_path)})]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        # If documents are already small enough, return as is
        if all(len(doc.page_content) <= self.chunk_size for doc in documents):
            return documents
        
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            # Fallback: manual splitting
            print(f"Warning: Using fallback text splitting due to error: {e}")
            return self._manual_split_documents(documents)
    
    def _manual_split_documents(self, documents: List[Document]) -> List[Document]:
        """Manual fallback text splitting."""
        chunks = []
        
        for doc in documents:
            text = doc.page_content
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                
                # Try to find a good break point
                if end < len(text):
                    # Look for sentence endings
                    for i in range(end, max(start, end - 100), -1):
                        if text[i] in '.!?\n':
                            end = i + 1
                            break
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata["chunk_id"] = len(chunks)
                    chunk_metadata["start_char"] = start
                    chunk_metadata["end_char"] = end
                    
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
                
                start = end
                
                # Prevent infinite loops
                if start >= len(text):
                    break
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate a summary of document characteristics.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document summary information
        """
        if not documents:
            return {"error": "No documents provided"}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Get unique sources
        sources = list(set(doc.metadata.get("source", "unknown") for doc in documents))
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chunk_size": total_chars // len(documents) if documents else 0,
            "sources": sources,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    def validate_document(self, file_path: str) -> bool:
        """
        Validate if a document can be processed.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if document is valid, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            file_extension = file_path.suffix.lower()
            supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.html']
            
            if file_extension not in supported_extensions:
                return False
            
            # Check file size (max 100MB)
            if file_path.stat().st_size > 100 * 1024 * 1024:
                return False
            
            return True
            
        except Exception:
            return False