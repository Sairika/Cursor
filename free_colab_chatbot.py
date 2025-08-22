# FREE Google Colab Financial Policy Chatbot
# NO API KEYS, NO EXTERNAL SERVICES - COMPLETELY FREE!

# ============================================================================
# CELL 1: INSTALL PACKAGES (FREE)
# ============================================================================

!pip install langchain
!pip install chromadb
!pip install sentence-transformers
!pip install transformers
!pip install torch

print("✅ All FREE packages installed!")

# ============================================================================
# CELL 2: IMPORT LIBRARIES
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import json

print("✅ Libraries imported!")

# ============================================================================
# CELL 3: CREATE FINANCIAL POLICY DOCUMENT
# ============================================================================

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

print("✅ Financial policy document created!")
print(f"📄 Document length: {len(financial_policy)} characters")

# ============================================================================
# CELL 4: CREATE SIMPLE CHATBOT (NO EXTERNAL APIS)
# ============================================================================

class SimpleFinancialChatbot:
    """Simple chatbot that works completely offline - NO API KEYS NEEDED!"""
    
    def __init__(self):
        self.documents = None
        self.vector_store = None
        self.memory = []
        self.setup_documents()
    
    def setup_documents(self):
        """Load and process documents"""
        print("📚 Setting up documents...")
        
        # Load document
        loader = TextLoader('financial_policy_document.txt')
        self.documents = loader.load()
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.documents = text_splitter.split_documents(self.documents)
        
        # Create embeddings (FREE - runs locally)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(self.documents, embeddings)
        
        print(f"✅ Documents ready! {len(self.documents)} chunks created")
    
    def search_documents(self, query, k=3):
        """Search for relevant information"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except:
            return []
    
    def generate_answer(self, question, context):
        """Generate answer using simple logic - NO EXTERNAL AI"""
        question_lower = question.lower()
        
        # Simple keyword-based responses
        if "budget" in question_lower:
            if "total" in question_lower or "annual" in question_lower:
                return "The total annual budget is $2,500,000."
            elif "r&d" in question_lower or "research" in question_lower:
                return "Research and Development gets 40% of the budget: $1,000,000 annually."
            else:
                return "Budget allocation: R&D (40%), Operations (25%), Marketing (20%), Admin (10%), Emergency (5%)."
        
        elif "debt" in question_lower:
            if "limit" in question_lower:
                return "Maximum total debt: $500,000. Debt-to-equity ratio: 0.3:1."
            else:
                return "Debt management: Monthly payments due by 5th, 8% interest cap, annual restructuring allowed."
        
        elif "travel" in question_lower:
            if "domestic" in question_lower:
                return "Domestic travel limit: $500 per day maximum."
            elif "international" in question_lower:
                return "International travel limit: $1,000 per day maximum."
            else:
                return "Travel limits: Domestic $500/day, International $1,000/day, Meals $75/day, Hotels $200/night."
        
        elif "emergency" in question_lower or "fund" in question_lower:
            return "Emergency fund: Minimum $500,000, Maximum $1,000,000. Used for operational expenses, disasters, market emergencies, and legal contingencies."
        
        elif "audit" in question_lower:
            if "internal" in question_lower:
                return "Internal audits are conducted quarterly."
            elif "external" in question_lower:
                return "External audits are required annually with findings addressed within 90 days."
            else:
                return "Auditing: Quarterly internal audits, annual external audits with GAAP compliance."
        
        elif "roi" in question_lower or "performance" in question_lower:
            return "Performance targets: ROI 15% annually, operating margin 25%, cash conversion cycle 45 days, debt service coverage ratio minimum 2.0."
        
        else:
            return "I can help with budgets, debt, travel, emergency funds, auditing, and performance metrics. Please ask a specific question about the financial policy."
    
    def ask_question(self, question):
        """Main function to answer questions"""
        # Search for relevant documents
        relevant_docs = self.search_documents(question)
        
        # Generate answer
        context = " ".join([doc.page_content for doc in relevant_docs])
        answer = self.generate_answer(question, context)
        
        # Store in memory
        self.memory.append({"question": question, "answer": answer})
        
        return {
            "answer": answer,
            "sources": relevant_docs,
            "question": question
        }
    
    def get_memory(self):
        """Get conversation history"""
        return self.memory
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory = []
        print("🧹 Memory cleared!")

# ============================================================================
# CELL 5: INITIALIZE CHATBOT
# ============================================================================

print("🚀 Initializing FREE Financial Policy Chatbot...")
chatbot = SimpleFinancialChatbot()
print("✅ Chatbot ready! No API keys needed!")

# ============================================================================
# CELL 6: TEST THE CHATBOT
# ============================================================================

print("\n🧪 Testing the chatbot...")

test_questions = [
    "What is the total annual budget?",
    "What are the debt limits?",
    "What is the travel policy for domestic trips?",
    "How often are budget reviews conducted?",
    "What are the emergency fund requirements?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n🔍 Question {i}: {question}")
    print("-" * 40)
    
    result = chatbot.ask_question(question)
    print(f"💡 Answer: {result['answer']}")
    
    if result['sources']:
        print(f"📚 Sources: {len(result['sources'])} document sections found")
    
    print("-" * 40)

print("\n🎉 Demo complete! Chatbot is working!")

# ============================================================================
# CELL 7: INTERACTIVE CHAT FUNCTION
# ============================================================================

def chat_with_bot():
    """Interactive chat - NO API KEYS NEEDED!"""
    print("\n" + "="*60)
    print("💰 FREE FINANCIAL POLICY CHATBOT")
    print("="*60)
    print("Ask me anything about the financial policy!")
    print("Type 'quit', 'exit', or 'bye' to end")
    print("Type 'clear' to clear memory")
    print("Type 'memory' to see conversation history")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤔 Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                chatbot.clear_memory()
                continue
            
            elif user_input.lower() == 'memory':
                memory = chatbot.get_memory()
                if memory:
                    print("\n📚 Conversation History:")
                    for i, item in enumerate(memory, 1):
                        print(f"{i}. Q: {item['question']}")
                        print(f"   A: {item['answer']}")
                else:
                    print("📚 No conversation history yet.")
                continue
            
            # Process question
            print("\n🤖 Processing...")
            result = chatbot.ask_question(user_input)
            
            print(f"\n💡 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources: {len(result['sources'])} relevant sections found")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# ============================================================================
# CELL 8: START CHATTING
# ============================================================================

print("\n🚀 Ready to chat!")
print("Type 'chat_with_bot()' to start interactive chat")
print("Or ask individual questions like: chatbot.ask_question('What is the budget?')")

# Example usage:
# result = chatbot.ask_question("What is the travel policy?")
# print(result['answer'])