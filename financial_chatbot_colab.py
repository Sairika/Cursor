# Google Colab Notebook for Financial Policy Chatbot
# Join Venture AI (JVAI) - AI Developer Assessment
# 
# To use this in Google Colab:
# 1. Copy this code to a new Colab notebook
# 2. Run each section as separate cells
# 3. Follow the instructions in each section

# ============================================================================
# SECTION 1: INSTALLATION AND SETUP
# ============================================================================

# Install required packages
!pip install --user "ibm-watsonx-ai==0.2.6"
!pip install --user "langchain==0.1.16"
!pip install --user "langchain-ibm==0.1.4"
!pip install --user "transformers==4.41.2"
!pip install --user "huggingface-hub==0.23.4"
!pip install --user "sentence-transformers==2.5.1"
!pip install --user "chromadb"
!pip install --user "wget==3.2"
!pip install --user --upgrade torch --index-url https://download.pytorch.org/whl/cpu

print("✅ All packages installed successfully!")

# ============================================================================
# SECTION 2: IMPORT LIBRARIES
# ============================================================================

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
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

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass

print("✅ All libraries imported successfully!")

# ============================================================================
# SECTION 3: CREATE FINANCIAL POLICY DOCUMENT
# ============================================================================

# Create the financial policy document
financial_policy = """FINANCIAL POLICY DOCUMENT
Join Venture AI (JVAI)
Version 1.0 - Effective Date: January 2025

1. BUDGET ALLOCATION AND MANAGEMENT

1.1 Annual Budget Overview
- Total annual budget: $2,500,000
- Research and Development: 40% ($1,000,000)
- Operations and Infrastructure: 25% ($625,000)
- Marketing and Sales: 20% ($500,000)
- Administrative Costs: 10% ($250,000)
- Emergency Reserve: 5% ($125,000)

1.2 Budget Approval Process
- All expenses above $10,000 require VP approval
- Department heads can approve expenses up to $5,000
- Monthly budget reviews conducted on the 15th of each month
- Quarterly budget adjustments allowed with board approval

2. DEBT MANAGEMENT POLICY

2.1 Debt Limits
- Maximum total debt: $500,000
- Maximum debt-to-equity ratio: 0.3:1
- Preferred debt instruments: Convertible notes and term loans
- Minimum credit rating requirement: BBB

2.2 Debt Servicing
- Monthly debt service payments due by the 5th of each month
- Interest rate cap: 8% annually
- Early repayment penalties: 2% of remaining balance
- Debt restructuring allowed once per fiscal year

3. INFRASTRUCTURE INVESTMENT

3.1 Technology Infrastructure
- Cloud computing budget: $300,000 annually
- Hardware refresh cycle: Every 3 years
- Software licensing: $150,000 annually
- Cybersecurity investment: $100,000 annually

3.2 Office Infrastructure
- Office space budget: $200,000 annually
- Equipment and furniture: $75,000 annually
- Maintenance and utilities: $50,000 annually

4. EXPENSE POLICIES

4.1 Travel and Entertainment
- Domestic travel: $500 per day maximum
- International travel: $1,000 per day maximum
- Meal expenses: $75 per day maximum
- Hotel accommodation: $200 per night maximum

4.2 Procurement
- Three quotes required for purchases above $5,000
- Preferred vendor program for office supplies
- Bulk purchasing discounts for items above $10,000
- Equipment leasing preferred over purchasing for items above $50,000

5. FINANCIAL REPORTING

5.1 Monthly Reports
- Cash flow statement due by 10th of each month
- Budget variance analysis due by 15th of each month
- Department spending summaries due by 20th of each month

5.2 Quarterly Reports
- Comprehensive financial statements
- Budget performance review
- Risk assessment and mitigation strategies
- Board presentation materials

6. COMPLIANCE AND AUDITING

6.1 Internal Controls
- Segregation of duties for all financial transactions
- Dual approval required for payments above $25,000
- Monthly bank reconciliations
- Quarterly internal audits

6.2 External Auditing
- Annual external audit required
- Audit findings must be addressed within 90 days
- Compliance with GAAP standards mandatory
- Tax filing deadlines strictly enforced

7. INVESTMENT POLICY

7.1 Investment Guidelines
- Maximum 20% of cash reserves in marketable securities
- Investment grade bonds preferred
- Diversification across asset classes required
- Maximum single investment: 5% of total portfolio

7.2 Risk Management
- Maximum portfolio risk: 15% annual volatility
- Regular portfolio rebalancing (quarterly)
- Stop-loss orders for equity positions
- Currency hedging for international investments

8. EMERGENCY FUNDS

8.1 Reserve Requirements
- Minimum emergency fund: $500,000
- Maximum emergency fund: $1,000,000
- Emergency fund can only be used for:
  * Unforeseen operational expenses
  * Natural disasters
  * Market emergencies
  * Legal contingencies

8.2 Emergency Fund Management
- Monthly review of emergency fund adequacy
- Investment in high-liquidity instruments only
- Board approval required for emergency fund usage
- Replenishment plan must be submitted within 30 days of usage

9. PERFORMANCE METRICS

9.1 Key Performance Indicators
- Return on Investment (ROI): Target 15% annually
- Operating margin: Target 25%
- Cash conversion cycle: Target 45 days
- Debt service coverage ratio: Minimum 2.0

9.2 Reporting Frequency
- Daily: Cash position and daily transactions
- Weekly: Budget vs. actual spending
- Monthly: Comprehensive financial analysis
- Quarterly: Strategic financial review

10. POLICY UPDATES AND AMENDMENTS

10.1 Review Process
- Annual policy review required
- Changes require board approval
- Employee notification within 30 days of changes
- Training on new policies within 60 days

10.2 Version Control
- All policy changes tracked with version numbers
- Change log maintained for audit purposes
- Previous versions archived for reference
- Digital signatures required for policy approvals

END OF DOCUMENT"""

# Save to file
with open('financial_policy_document.txt', 'w') as f:
    f.write(financial_policy)

print("✅ Financial policy document created successfully!")
print(f"📄 Document length: {len(financial_policy)} characters")
print("📁 Saved as: financial_policy_document.txt")

# ============================================================================
# SECTION 4: CHATBOT IMPLEMENTATION
# ============================================================================

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

print("✅ Chatbot class implemented successfully!")

# ============================================================================
# SECTION 5: INITIALIZE THE CHATBOT
# ============================================================================

# Initialize the chatbot
try:
    chatbot = FinancialPolicyChatbot()
    print("\n🎉 Chatbot is ready to answer your questions!")
except Exception as e:
    print(f"\n❌ Failed to initialize chatbot: {e}")
    print("Please check the error message above and ensure all packages are installed correctly.")

# ============================================================================
# SECTION 6: DEMO QUESTIONS
# ============================================================================

# Test the chatbot with sample questions
if 'chatbot' in locals():
    demo_questions = [
        "What is the total annual budget?",
        "What are the debt limits?",
        "What is the travel policy for domestic trips?",
        "How often are budget reviews conducted?",
        "What are the emergency fund requirements?"
    ]
    
    print("\n🧪 Testing chatbot with sample questions...\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"🔍 Question {i}: {question}")
        print("-" * 50)
        
        try:
            result = chatbot.ask_question(question)
            print(f"💡 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"📚 Sources: {len(result['sources'])} document sections found")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50 + "\n")
else:
    print("❌ Chatbot not initialized. Please run the previous cell first.")

# ============================================================================
# SECTION 7: INTERACTIVE CHAT
# ============================================================================

# Interactive chat function
def interactive_chat():
    """Interactive chat interface for the chatbot"""
    if 'chatbot' not in locals():
        print("❌ Chatbot not initialized. Please run the initialization cell first.")
        return
    
    print("\n" + "="*60)
    print("💰 FINANCIAL POLICY CHATBOT - INTERACTIVE MODE")
    print("="*60)
    print("Ask me anything about the financial policy document!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'clear' to clear conversation memory")
    print("Type 'search <query>' to search for specific information")
    print("="*60)
    
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

# Start interactive chat
print("\n🚀 Starting interactive chat mode...")
print("You can now ask questions about the financial policy!")
print("Type 'interactive_chat()' in the next cell to start chatting.")

# ============================================================================
# SECTION 8: CONVERSATION MEMORY TEST
# ============================================================================

# Test conversation memory
if 'chatbot' in locals():
    print("\n🧠 Testing conversation memory...\n")
    
    # First question
    print("🔍 Question 1: What is the total annual budget?")
    result1 = chatbot.ask_question("What is the total annual budget?")
    print(f"💡 Answer: {result1['answer']}\n")
    
    # Follow-up question (should use context from previous question)
    print("🔍 Question 2: How much is allocated to R&D?")
    result2 = chatbot.ask_question("How much is allocated to R&D?")
    print(f"💡 Answer: {result2['answer']}\n")
    
    # Another follow-up question
    print("🔍 Question 3: What about marketing and sales?")
    result3 = chatbot.ask_question("What about marketing and sales?")
    print(f"💡 Answer: {result3['answer']}\n")
    
    # Check conversation history
    print("📚 Current conversation history:")
    history = chatbot.get_conversation_history()
    for i, msg in enumerate(history, 1):
        print(f"   {i}. {msg.content[:100]}..." if len(msg.content) > 100 else f"   {i}. {msg.content}")
    
    # Clear memory
    print("\n🧹 Clearing conversation memory...")
    chatbot.clear_memory()
    print("✅ Memory cleared!")
    
else:
    print("❌ Chatbot not initialized. Please run the initialization cell first.")

# ============================================================================
# SECTION 9: DOCUMENT SEARCH TEST
# ============================================================================

# Test document search functionality
if 'chatbot' in locals():
    print("\n🔍 Testing document search functionality...\n")
    
    search_queries = [
        "budget approval process",
        "debt management",
        "travel expenses",
        "emergency funds",
        "compliance requirements"
    ]
    
    for query in search_queries:
        print(f"🔍 Searching for: {query}")
        print("-" * 40)
        
        try:
            docs = chatbot.search_documents(query, k=2)
            print(f"Found {len(docs)} relevant document chunks:")
            
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Chunk {i} ---")
                content = doc.page_content
                if len(content) > 200:
                    content = content[:200] + "..."
                print(content)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40 + "\n")
else:
    print("❌ Chatbot not initialized. Please run the initialization cell first.")

# ============================================================================
# SECTION 10: CUSTOM QUESTIONS
# ============================================================================

# Ask your own questions here
if 'chatbot' in locals():
    # Example questions - feel free to modify these or add your own
    custom_questions = [
        "What are the procurement requirements for purchases above $5,000?",
        "How often are internal audits conducted?",
        "What is the maximum portfolio risk allowed?",
        "Can you explain the emergency fund usage policy?",
        "What are the key performance indicators?"
    ]
    
    print("\n❓ Custom Questions Demo\n")
    
    for i, question in enumerate(custom_questions, 1):
        print(f"🔍 Question {i}: {question}")
        print("-" * 50)
        
        try:
            result = chatbot.ask_question(question)
            print(f"💡 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"📚 Sources: {len(result['sources'])} document sections found")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50 + "\n")
else:
    print("❌ Chatbot not initialized. Please run the initialization cell first.")

print("\n🎉 Financial Policy Chatbot Demo Complete!")
print("You can now:")
print("1. Ask questions using the chatbot")
print("2. Test conversation memory")
print("3. Search for specific information")
print("4. Start interactive chat mode")
print("\nType 'interactive_chat()' to start chatting!")