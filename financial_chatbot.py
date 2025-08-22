"""
AI-Powered Financial Policy Chatbot
Join Venture AI (JVAI) - AI Developer Assessment

This chatbot can answer questions about the financial policy document using:
- Vector search for document retrieval
- Conversation memory for context awareness
- LLM-powered responses based on retrieved information

Author: [Your Name]
Date: [Current Date]
"""

import os
import warnings
from typing import List, Dict, Any
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import required libraries
try:
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
    from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages using:")
    print("pip install langchain langchain-ibm ibm-watsonx-ai transformers sentence-transformers chromadb")
    exit(1)

@dataclass
class ChatbotConfig:
    """Configuration class for the chatbot"""
    document_path: str = "financial_policy_document.txt"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_id: str = "google/flan-t5-xl"
    max_tokens: int = 256
    temperature: float = 0.3
    search_k: int = 3  # Number of document chunks to retrieve

class FinancialPolicyChatbot:
    """
    AI-powered chatbot for answering questions about financial policy documents.
    
    Features:
    - Document ingestion and chunking
    - Vector search using ChromaDB
    - Conversation memory for context awareness
    - LLM-powered responses with source tracking
    """
    
    def __init__(self, config: ChatbotConfig = None):
        """Initialize the chatbot with configuration"""
        self.config = config or ChatbotConfig()
        self.documents = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        
        print("🚀 Initializing Financial Policy Chatbot...")
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
            
            # Step 3: Initialize the language model
            print("🤖 Initializing language model...")
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
            # Initialize embeddings model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
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
        """Initialize the language model"""
        try:
            # Set up model parameters
            parameters = {
                GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                GenParams.MAX_NEW_TOKENS: self.config.max_tokens,
                GenParams.TEMPERATURE: self.config.temperature
            }
            
            # Set up credentials (using skills-network for demo)
            credentials = {
                "url": "https://us-south.ml.cloud.ibm.com"
            }
            
            project_id = "skills-network"
            
            # Initialize the model
            model = Model(
                model_id=self.config.model_id,
                params=parameters,
                credentials=credentials,
                project_id=project_id
            )
            
            # Create LangChain wrapper
            self.llm = WatsonxLLM(model=model)
            
            print(f"🤖 Language model {self.config.model_id} initialized")
            
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
            
            # Create custom prompt template
            prompt_template = """You are a helpful AI assistant that answers questions about financial policies. 
            Use the following context to answer the question. If you don't know the answer, say "I don't have enough information to answer that question."

            Context: {context}

            Question: {question}

            Answer based on the context provided:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create the conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.search_k}
                ),
                memory=self.memory,
                get_chat_history=lambda h: h,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT}
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
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            # Get response from the chain
            result = self.qa_chain({"question": question})
            
            return {
                "answer": result.get("answer", "No answer generated"),
                "sources": result.get("source_documents", []),
                "question": question
            }
            
        except Exception as e:
            print(f"❌ Error asking question: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "sources": [],
                "question": question
            }
    
    def get_conversation_history(self) -> List[tuple]:
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
    print("💰 FINANCIAL POLICY CHATBOT")
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
    print("🚀 Starting Financial Policy Chatbot...")
    
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