"""
Financial Policy Document Chatbot
A RAG-powered chatbot that can answer questions about financial policy documents
using LangChain, sentence transformers, and open-source models.

Author: AI Developer Assessment
Date: 2025
"""

import os
import logging
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.schema import Document

# For PDF processing
import PyPDF2
import io

# For embeddings and vector search
import chromadb
from sentence_transformers import SentenceTransformer

# For conversation management
import json
from datetime import datetime

class FinancialPolicyChatbot:
    """
    A chatbot that can answer questions about financial policy documents using RAG.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the chatbot with the specified embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.documents = []
        self.chat_history = []
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Financial Policy Chatbot...")
        
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document (PDF or text) and convert it to LangChain Document objects.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        try:
            if file_path.lower().endswith('.pdf'):
                self.logger.info(f"Loading PDF document: {file_path}")
                return self._load_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                self.logger.info(f"Loading text document: {file_path}")
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading document: {e}")
            raise
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document and extract text."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                documents = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        # Create document with metadata including page number
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'page': page_num + 1,
                                'file_type': 'pdf'
                            }
                        )
                        documents.append(doc)
                        
                self.logger.info(f"Loaded {len(documents)} pages from PDF")
                return documents
                
        except Exception as e:
            self.logger.error(f"Error loading PDF: {e}")
            raise
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
            # Split text into chunks if it's very long
            if len(text) > 10000:
                chunks = self._split_text(text)
                documents = []
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': file_path,
                            'chunk': i + 1,
                            'file_type': 'text'
                        }
                    )
                    documents.append(doc)
                return documents
            else:
                return [Document(
                    page_content=text,
                    metadata={
                        'source': file_path,
                        'file_type': 'text'
                    }
                )]
                
        except Exception as e:
            self.logger.error(f"Error loading text file: {e}")
            raise
    
    def _split_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Split long text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    end = start + break_point + 1
                    chunk = text[start:end]
            
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    def process_documents(self, documents: List[Document], chunk_size: int = 1000, overlap: int = 200):
        """
        Process documents by splitting them into chunks and creating embeddings.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each text chunk
            overlap: Overlap between chunks
        """
        self.logger.info("Processing documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(self.documents)} chunks from documents")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            self.documents, 
            self.embeddings,
            collection_name="financial_policies"
        )
        
        self.logger.info("Vector store created successfully")
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create QA chain
        self._create_qa_chain()
        
    def _create_qa_chain(self):
        """Create the question-answering chain."""
        try:
            # Custom prompt template for financial policy questions
            prompt_template = """You are a helpful AI assistant that answers questions about financial policy documents. 
            Use the following context to answer the question. If you cannot find the answer in the context, 
            say "I don't have enough information to answer that question based on the provided documents."
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create the conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self._get_llm(),
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False
            )
            
            self.logger.info("QA chain created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating QA chain: {e}")
            raise
    
    def _get_llm(self):
        """
        Get an LLM instance. For Google Colab, we'll use a simple approach
        that can work with the available resources.
        """
        try:
            # Try to use HuggingFace Hub with a free model
            # You can replace this with your HuggingFace token if you have one
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_dummy_token"
            
            # Use a small, efficient model that works well on Colab
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-small",
                model_kwargs={"temperature": 0.5, "max_length": 512},
                huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            )
            
            return llm
            
        except Exception as e:
            self.logger.warning(f"Could not initialize HuggingFace Hub LLM: {e}")
            # Fallback to a simple text-based approach
            return self._create_simple_llm()
    
    def _create_simple_llm(self):
        """Create a simple LLM that works without external API calls."""
        class SimpleLLM:
            def __call__(self, prompt):
                # Simple keyword-based responses for demonstration
                # In a real implementation, you'd want a proper LLM
                return "This is a placeholder response. Please configure a proper LLM for better results."
        
        return SimpleLLM()
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer based on the loaded documents.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("Documents not processed yet. Call process_documents() first.")
        
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Get response from QA chain
            response = self.qa_chain({"question": question})
            
            # Extract answer and sources
            answer = response.get("answer", "No answer generated")
            source_docs = response.get("source_documents", [])
            
            # Format source information
            sources = []
            for doc in source_docs:
                source_info = {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                }
                sources.append(source_info)
            
            # Store in chat history
            chat_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'sources': sources
            }
            self.chat_history.append(chat_entry)
            
            return {
                'answer': answer,
                'sources': sources,
                'timestamp': chat_entry['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history."""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
        if self.memory:
            self.memory.clear()
        self.logger.info("Chat history cleared")
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents based on a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Documents not processed yet. Call process_documents() first.")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded documents.
        
        Returns:
            Dictionary containing document summary information
        """
        if not self.documents:
            return {"error": "No documents loaded"}
        
        try:
            total_chunks = len(self.documents)
            total_characters = sum(len(doc.page_content) for doc in self.documents)
            
            # Get unique sources
            sources = set()
            for doc in self.documents:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            return {
                'total_chunks': total_chunks,
                'total_characters': total_characters,
                'sources': list(sources),
                'average_chunk_size': total_characters / total_chunks if total_chunks > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

def main():
    """Main function to demonstrate the chatbot."""
    print("=== Financial Policy Document Chatbot ===\n")
    
    # Initialize chatbot
    chatbot = FinancialPolicyChatbot()
    
    # Example usage (you'll need to provide your own PDF file)
    print("To use this chatbot:")
    print("1. Place your financial policy PDF in the same directory")
    print("2. Update the file_path variable below")
    print("3. Run the chatbot\n")
    
    # Example file path - update this with your actual PDF file
    file_path = "financial_policy.pdf"  # Update this with your file name
    
    if os.path.exists(file_path):
        try:
            # Load and process documents
            print(f"Loading document: {file_path}")
            documents = chatbot.load_document(file_path)
            
            print("Processing documents...")
            chatbot.process_documents(documents)
            
            print("Chatbot ready! You can now ask questions about your financial policy document.")
            print("Type 'quit' to exit.\n")
            
            # Interactive chat loop
            while True:
                question = input("Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if question:
                    response = chatbot.ask_question(question)
                    print(f"\nAnswer: {response['answer']}")
                    
                    if response['sources']:
                        print("\nSources:")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"{i}. {source['metadata']}")
                            print(f"   Content: {source['content']}\n")
                    else:
                        print("\nNo specific sources found.\n")
                        
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your file and try again.")
    else:
        print(f"File not found: {file_path}")
        print("Please place your financial policy PDF in the same directory and update the file_path variable.")

if __name__ == "__main__":
    main()