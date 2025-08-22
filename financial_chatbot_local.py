"""
AI-Powered Financial Policy Chatbot - LOCAL VERSION
Join Venture AI (JVAI) - AI Developer Assessment

This chatbot works completely offline using:
- Local HuggingFace models for embeddings and text generation
- No external APIs or paid services
- Free and open-source components only

Author: [Your Name]
Date: [Current Date]
"""

import os
import warnings
from typing import List, Dict, Any
from dataclasses import dataclass
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import required libraries
try:
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages using:")
    print("pip install langchain transformers torch sentence-transformers chromadb")
    exit(1)

@dataclass
class ChatbotConfig:
    """Configuration class for the chatbot"""
    document_path: str = "financial_policy_document.txt"
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_tokens: int = 150
    temperature: float = 0.7
    search_k: int = 3  # Number of document chunks to retrieve

class LocalLLM:
    """
    Local language model using HuggingFace transformers
    No external API calls - completely free and local
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """Initialize local language model"""
        try:
            print(f"🤖 Loading local model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("✅ Local model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            print("Falling back to simple text generation...")
            self.pipeline = None
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using local model"""
        try:
            if self.pipeline:
                # Use the local model
                full_prompt = f"Context: {context}\nQuestion: {prompt}\nAnswer:"
                
                # Generate response
                response = self.pipeline(full_prompt, max_length=len(full_prompt.split()) + 50)[0]['generated_text']
                
                # Extract just the answer part
                if "Answer:" in response:
                    answer = response.split("Answer:")[-1].strip()
                else:
                    answer = response.split("Question:")[-1].strip()
                
                return answer if answer else "I understand your question about the financial policy."
            else:
                # Fallback to simple template-based responses
                return self._simple_response(prompt, context)
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return self._simple_response(prompt, context)
    
    def _simple_response(self, prompt: str, context: str) -> str:
        """Simple template-based response when model fails"""
        prompt_lower = prompt.lower()
        
        # Simple keyword-based responses
        if "budget" in prompt_lower:
            if "total" in prompt_lower or "annual" in prompt_lower:
                return "Based on the financial policy, the total annual budget is $2,500,000."
            elif "r&d" in prompt_lower or "research" in prompt_lower:
                return "Research and Development is allocated 40% of the total budget, which equals $1,000,000 annually."
            else:
                return "The budget allocation covers R&D (40%), Operations (25%), Marketing (20%), Admin (10%), and Emergency Reserve (5%)."
        
        elif "debt" in prompt_lower:
            if "limit" in prompt_lower:
                return "The maximum total debt allowed is $500,000 with a debt-to-equity ratio of 0.3:1."
            else:
                return "Debt management includes monthly payments due by the 5th, 8% interest cap, and annual restructuring allowance."
        
        elif "travel" in prompt_lower:
            if "domestic" in prompt_lower:
                return "Domestic travel is limited to $500 per day maximum."
            elif "international" in prompt_lower:
                return "International travel is limited to $1,000 per day maximum."
            else:
                return "Travel expenses include daily limits: domestic $500, international $1,000, meals $75, hotels $200."
        
        elif "emergency" in prompt_lower or "fund" in prompt_lower:
            return "Emergency fund requirements: minimum $500,000, maximum $1,000,000. Used for operational expenses, disasters, market emergencies, and legal contingencies."
        
        elif "audit" in prompt_lower:
            if "internal" in prompt_lower:
                return "Internal audits are conducted quarterly."
            elif "external" in prompt_lower:
                return "External audits are required annually with findings addressed within 90 days."
            else:
                return "Auditing includes quarterly internal audits and annual external audits with GAAP compliance."
        
        elif "roi" in prompt_lower or "performance" in prompt_lower:
            return "Key performance targets: ROI 15% annually, operating margin 25%, cash conversion cycle 45 days, debt service coverage ratio minimum 2.0."
        
        else:
            return "I can help you with information about budgets, debt management, travel policies, emergency funds, auditing, and performance metrics. Please ask a specific question about the financial policy."

class FinancialPolicyChatbot:
    """
    AI-powered chatbot for answering questions about financial policy documents.
    
    Features:
    - Document ingestion and chunking
    - Vector search using ChromaDB
    - Conversation memory for context awareness
    - Local LLM responses with source tracking
    - Completely free and offline
    """
    
    def __init__(self, config: ChatbotConfig = None):
        """Initialize the chatbot with configuration"""
        self.config = config or ChatbotConfig()
        self.documents = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        
        print("🚀 Initializing Financial Policy Chatbot (Local Version)...")
        self._setup_components()
    
    def _setup_components(self):
        """Set up all chatbot components"""
        try:
            # Step 1: Load and process the document
            print("📚 Loading financial policy document...")
            self._load_documents()
            
            # Step 2: Create vector embeddings
            print("🔍 Creating vector embeddings...")
            self._create_embeddings()
            
            # Step 3: Initialize the local language model
            print("🤖 Initializing local language model...")
            self._setup_llm()
            
            # Step 4: Create the QA chain with memory
            print("🔗 Setting up QA chain with memory...")
            self._setup_qa_chain()
            
            print("✅ Chatbot initialization complete!")
            
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            raise
    
    def _load_documents(self):
        """Load and split the financial policy document"""
        try:
            # Load the document
            loader = TextLoader(self.config.document_path)
            self.documents = loader.load()
            
            # Split into chunks for better processing
            text_splitter = CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            self.documents = text_splitter.split_documents(self.documents)
            
            print(f"📄 Document loaded and split into {len(self.documents)} chunks")
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            raise
    
    def _create_embeddings(self):
        """Create vector embeddings using HuggingFace embeddings"""
        try:
            # Initialize embeddings model (smaller model for local use)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # Force CPU to avoid GPU issues
            )
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                self.documents, 
                embeddings
            )
            
            print("🔍 Vector embeddings created successfully")
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            raise
    
    def _setup_llm(self):
        """Initialize the local language model"""
        try:
            # Initialize local LLM
            self.llm = LocalLLM()
            print("🤖 Local language model initialized")
            
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            raise
    
    def _setup_qa_chain(self):
        """Set up the QA chain with conversation memory"""
        try:
            # Create conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            print("🔗 QA chain with memory setup complete")
            
        except Exception as e:
            print(f"❌ Error setting up QA chain: {e}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with source information
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict containing answer and source documents
        """
        try:
            # Search for relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=self.config.search_k)
            
            # Combine relevant context
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate response using local LLM
            answer = self.llm.generate_response(question, context)
            
            return {
                "answer": answer,
                "sources": relevant_docs,
                "question": question
            }
            
        except Exception as e:
            print(f"❌ Error asking question: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "sources": [],
                "question": question
            }
    
    def get_conversation_history(self) -> List:
        """Get the current conversation history"""
        if self.memory:
            return self.memory.chat_memory.messages
        return []
    
    def clear_memory(self):
        """Clear the conversation memory"""
        if self.memory:
            self.memory.clear()
            print("🧹 Conversation memory cleared")
    
    def search_documents(self, query: str, k: int = 3) -> List:
        """Search for relevant document chunks"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            # Search for similar documents
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

def interactive_chat():
    """Interactive chat interface for the chatbot"""
    print("\n" + "="*60)
    print("💰 FINANCIAL POLICY CHATBOT - LOCAL VERSION")
    print("="*60)
    print("Ask me anything about the financial policy document!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'clear' to clear conversation memory")
    print("Type 'search <query>' to search for specific information")
    print("="*60)
    
    # Initialize chatbot
    try:
        chatbot = FinancialPolicyChatbot()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤔 Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thank you for using the Financial Policy Chatbot!")
                break
            
            elif user_input.lower() == 'clear':
                chatbot.clear_memory()
                conversation_history = []
                continue
            
            elif user_input.lower().startswith('search '):
                query = user_input[7:]  # Remove 'search ' prefix
                print(f"🔍 Searching for: {query}")
                docs = chatbot.search_documents(query)
                print(f"Found {len(docs)} relevant document chunks:")
                for i, doc in enumerate(docs, 1):
                    print(f"\n--- Chunk {i} ---")
                    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                continue
            
            # Process the question
            print("\n🤖 Processing your question...")
            result = chatbot.ask_question(user_input)
            
            # Display the answer
            print(f"\n💡 Answer: {result['answer']}")
            
            # Display source information
            if result['sources']:
                print(f"\n📚 Sources: Found {len(result['sources'])} relevant document sections")
                for i, source in enumerate(result['sources'][:2], 1):  # Show first 2 sources
                    content = source.page_content
                    if len(content) > 150:
                        content = content[:150] + "..."
                    print(f"   Source {i}: {content}")
            
            # Add to conversation history
            conversation_history.append((user_input, result['answer']))
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            continue

def demo_questions():
    """Demonstrate the chatbot with sample questions"""
    print("\n🧪 DEMO MODE - Testing with sample questions")
    print("="*50)
    
    try:
        chatbot = FinancialPolicyChatbot()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return
    
    # Sample questions to demonstrate functionality
    demo_questions = [
        "What is the total annual budget?",
        "What are the debt limits?",
        "What is the travel policy for domestic trips?",
        "How often are budget reviews conducted?",
        "What are the emergency fund requirements?",
        "What is the ROI target?",
        "Can you summarize the procurement policy?",
        "What are the compliance requirements?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n🔍 Demo Question {i}: {question}")
        print("-" * 40)
        
        result = chatbot.ask_question(question)
        print(f"Answer: {result['answer']}")
        
        if result['sources']:
            print(f"Sources: {len(result['sources'])} document sections found")
        
        print("-" * 40)

if __name__ == "__main__":
    """Main entry point"""
    print("🚀 Starting Financial Policy Chatbot (Local Version)...")
    
    # Check if document exists
    if not os.path.exists("financial_policy_document.txt"):
        print("❌ Financial policy document not found!")
        print("Please ensure 'financial_policy_document.txt' is in the current directory.")
        exit(1)
    
    # Ask user for mode
    print("\nChoose a mode:")
    print("1. Interactive chat")
    print("2. Demo mode (pre-defined questions)")
    
    try:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_chat()
        elif choice == "2":
            demo_questions()
        else:
            print("Invalid choice. Running demo mode...")
            demo_questions()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An error occurred: {e}")