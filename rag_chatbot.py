"""
Main RAG Chatbot module that integrates document processing, vector storage, and LLM generation.
Provides a conversational interface for querying documents.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_interface import create_llm, BaseLLM

class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10, max_length: int = 1000):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of conversation turns to remember
            max_length: Maximum total length of conversation history
        """
        self.max_history = max_history
        self.max_length = max_length
        self.conversations = []
        self.current_conversation = []
    
    def add_exchange(self, question: str, answer: str, sources: List[Any] = None):
        """
        Add a question-answer exchange to memory.
        
        Args:
            question: User's question
            answer: Bot's answer
            sources: Source documents used for the answer
        """
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or []
        }
        
        self.current_conversation.append(exchange)
        
        # Maintain conversation length
        if len(self.current_conversation) > self.max_history:
            self.current_conversation.pop(0)
        
        # Check total length
        total_length = sum(len(ex["question"]) + len(ex["answer"]) for ex in self.current_conversation)
        while total_length > self.max_length and len(self.current_conversation) > 1:
            removed = self.current_conversation.pop(0)
            total_length -= len(removed["question"]) + len(removed["answer"])
    
    def get_context(self, max_turns: int = 3) -> str:
        """
        Get conversation context for the LLM.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        if not self.current_conversation:
            return ""
        
        # Get recent turns
        recent_turns = self.current_conversation[-max_turns:]
        
        context = "Recent conversation:\n"
        for turn in recent_turns:
            context += f"User: {turn['question']}\n"
            context += f"Assistant: {turn['answer']}\n\n"
        
        return context.strip()
    
    def clear_current(self):
        """Clear the current conversation."""
        if self.current_conversation:
            self.conversations.append(self.current_conversation.copy())
            self.current_conversation = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "current_conversation_length": len(self.current_conversation),
            "total_conversations": len(self.conversations),
            "max_history": self.max_history,
            "max_length": self.max_length
        }

class RAGChatbot:
    """Main RAG chatbot class that integrates all components."""
    
    def __init__(self, 
                 model_name: str = None,
                 vector_db_type: str = "faiss",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 **kwargs):
        """
        Initialize the RAG chatbot.
        
        Args:
            model_name: Name of the language model to use
            vector_db_type: Type of vector database ("faiss" or "chromadb")
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            **kwargs: Additional configuration parameters
        """
        # Create necessary directories
        Config.create_directories()
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = VectorStore(
            embedding_model=Config.EMBEDDING_MODEL,
            vector_db_type=vector_db_type,
            db_path=Config.VECTOR_DB_PATH
        )
        
        # Initialize LLM
        model_config = Config.get_model_config(model_name)
        self.llm = create_llm(
            model_name=model_name or Config.DEFAULT_MODEL,
            **model_config
        )
        
        # Initialize memory
        self.memory = ConversationMemory(
            max_history=Config.MEMORY_WINDOW,
            max_length=Config.MAX_HISTORY_LENGTH
        )
        
        # Chatbot state
        self.documents_loaded = False
        self.current_document = None
        
        print("RAG Chatbot initialized successfully!")
        print(f"Model: {self.llm.get_model_info().get('model_name', 'Unknown')}")
        print(f"Vector DB: {self.vector_store.vector_db_type}")
    
    def load_document(self, file_path: str) -> bool:
        """
        Load and process a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading document: {file_path}")
            
            # Validate document
            if not self.document_processor.validate_document(file_path):
                print("Invalid document format or file")
                return False
            
            # Load document
            documents = self.document_processor.load_document(file_path)
            print(f"Loaded {len(documents)} document(s)")
            
            # Split into chunks
            chunks = self.document_processor.split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            
            # Add to vector store
            success = self.vector_store.add_documents(chunks)
            if success:
                self.documents_loaded = True
                self.current_document = file_path
                
                # Save vector store
                self.vector_store.save()
                
                # Print document summary
                summary = self.document_processor.get_document_summary(chunks)
                print("Document Summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
                
                return True
            else:
                print("Failed to add documents to vector store")
                return False
                
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def ask(self, question: str, include_sources: bool = True) -> str:
        """
        Ask a question and get an answer.
        
        Args:
            question: User's question
            include_sources: Whether to include source information
            
        Returns:
            Generated answer
        """
        if not self.documents_loaded:
            return "No documents loaded. Please load a document first using load_document()."
        
        try:
            # Search for relevant documents
            search_results = self.vector_store.search(question, top_k=Config.SIMILARITY_TOP_K)
            
            if not search_results:
                return "I couldn't find any relevant information in the documents to answer your question."
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for doc, score in search_results:
                context_parts.append(doc.page_content)
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "similarity": f"{score:.3f}"
                })
            
            context = "\n\n".join(context_parts)
            
            # Get conversation context
            conversation_context = self.memory.get_context()
            
            # Create prompt
            if conversation_context:
                prompt = f"{Config.SYSTEM_PROMPT}\n\n{conversation_context}\n\n{Config.QA_PROMPT_TEMPLATE.format(context=context, question=question)}"
            else:
                prompt = f"{Config.SYSTEM_PROMPT}\n\n{Config.QA_PROMPT_TEMPLATE.format(context=context, question=question)}"
            
            # Generate answer
            answer = self.llm.generate(prompt)
            
            # Add source information if requested
            if include_sources and sources:
                answer += "\n\nSources:"
                for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                    answer += f"\n{i}. {source['source']} (similarity: {source['similarity']})"
                    answer += f"\n   {source['content']}"
            
            # Add to memory
            self.memory.add_exchange(question, answer, sources)
            
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(error_msg)
            return error_msg
    
    def summarize_document(self, max_length: int = 500) -> str:
        """
        Generate a summary of the loaded document.
        
        Args:
            max_length: Maximum length of the summary
            
        Returns:
            Document summary
        """
        if not self.documents_loaded:
            return "No documents loaded. Please load a document first."
        
        try:
            # Get all documents from vector store
            if self.vector_store.vector_db_type == "faiss":
                documents = self.vector_store.documents
            else:
                # For ChromaDB, we need to get all documents
                documents = []
                # This is a simplified approach - in practice, you might want to store
                # the original documents separately
                return "Summary not available for ChromaDB storage."
            
            if not documents:
                return "No documents available for summarization."
            
            # Combine all document content
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # Create summarization prompt
            prompt = f"{Config.SUMMARIZATION_PROMPT.format(context=full_content[:2000])}\n\nPlease provide a concise summary in {max_length} words or less."
            
            # Generate summary
            summary = self.llm.generate(prompt, max_length=max_length)
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the loaded document."""
        if not self.documents_loaded:
            return {"error": "No documents loaded"}
        
        try:
            vector_stats = self.vector_store.get_stats()
            memory_stats = self.memory.get_stats()
            
            return {
                "current_document": self.current_document,
                "vector_store": vector_stats,
                "memory": memory_stats,
                "llm_info": self.llm.get_model_info()
            }
        except Exception as e:
            return {"error": f"Could not get document info: {e}"}
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear_current()
        print("Conversation memory cleared.")
    
    def change_model(self, model_name: str) -> bool:
        """
        Change the language model.
        
        Args:
            model_name: Name of the new model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Changing model to: {model_name}")
            
            # Get configuration for the new model
            model_config = Config.get_model_config(model_name)
            
            # Create new LLM
            new_llm = create_llm(model_name=model_name, **model_config)
            
            if new_llm.is_available():
                self.llm = new_llm
                print(f"Successfully changed to model: {model_name}")
                return True
            else:
                print(f"Failed to load model: {model_name}")
                return False
                
        except Exception as e:
            print(f"Error changing model: {e}")
            return False
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        if not self.documents_loaded:
            print("No documents loaded. Please load a document first.")
            return
        
        print("\n" + "="*50)
        print("RAG Chatbot Interactive Session")
        print("="*50)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'summary' to get a document summary")
        print("Type 'info' to get system information")
        print("Type 'clear' to clear conversation memory")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! Ending chat session.")
                    break
                
                elif user_input.lower() == 'summary':
                    print("\nGenerating document summary...")
                    summary = self.summarize_document()
                    print(f"Summary: {summary}")
                
                elif user_input.lower() == 'info':
                    info = self.get_document_info()
                    print("\nSystem Information:")
                    print(json.dumps(info, indent=2, default=str))
                
                elif user_input.lower() == 'clear':
                    self.clear_memory()
                
                elif user_input:
                    print("\nThinking...")
                    answer = self.ask(user_input)
                    print(f"Assistant: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nChat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or type 'quit' to exit.")

def main():
    """Main function to run the chatbot."""
    print("RAG Chatbot - Document Q&A System")
    print("="*40)
    
    # Create chatbot instance
    chatbot = RAGChatbot()
    
    # Check if documents exist in the sample directory
    sample_dir = Path(Config.SAMPLE_DOCS_DIR)
    if sample_dir.exists() and any(sample_dir.iterdir()):
        print(f"\nFound sample documents in {sample_dir}")
        print("Available documents:")
        for doc in sample_dir.iterdir():
            if doc.is_file():
                print(f"  - {doc.name}")
        
        # Ask user if they want to load a sample document
        response = input("\nWould you like to load a sample document? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            doc_name = input("Enter document name: ").strip()
            doc_path = sample_dir / doc_name
            if doc_path.exists():
                if chatbot.load_document(str(doc_path)):
                    print("Document loaded successfully!")
                    chatbot.interactive_chat()
                else:
                    print("Failed to load document.")
            else:
                print("Document not found.")
    
    # If no sample documents or user doesn't want to load them
    if not chatbot.documents_loaded:
        print("\nTo get started:")
        print("1. Place your document in the sample_documents folder")
        print("2. Run: chatbot.load_document('path/to/your/document')")
        print("3. Start chatting with: chatbot.interactive_chat()")
        
        # Try to create a sample document
        sample_doc_path = sample_dir / "sample_policy.txt"
        sample_dir.mkdir(exist_ok=True)
        
        sample_content = """Sample Company Policy Document

1. Code of Conduct
Our company is committed to maintaining a workplace built on integrity, respect, and accountability.

2. Work Hours
Standard work hours are 9:00 AM to 5:00 PM, Monday through Friday.

3. Leave Policy
Employees are entitled to 20 vacation days per year, plus standard holidays.

4. Dress Code
Business casual attire is required in the office.

5. Internet Usage
Company internet is for work-related activities only. Limited personal use is permitted during breaks.
"""
        
        with open(sample_doc_path, 'w') as f:
            f.write(sample_content)
        
        print(f"\nCreated sample document: {sample_doc_path}")
        print("You can load it with: chatbot.load_document('sample_documents/sample_policy.txt')")

if __name__ == "__main__":
    main()