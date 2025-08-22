#!/usr/bin/env python3
"""
Financial Policy Chatbot Demo
This script demonstrates how to use the FinancialPolicyChatbot class
with a sample financial policy document.

Author: AI Developer Assessment
Date: 2025
"""

import os
import tempfile
from financial_chatbot import FinancialPolicyChatbot

def create_sample_policy():
    """Create a sample financial policy document for demonstration."""
    sample_text = """
    FINANCIAL POLICY DOCUMENT
    Company: Innovatech Solutions
    Effective Date: January 1, 2025
    Version: 2.0

    SECTION 1: BUDGET ALLOCATION
    Our annual budget allocation for 2025 is $2.5 million, distributed across the following areas:
    - Research and Development: $800,000 (32%)
    - Infrastructure: $600,000 (24%)
    - Marketing: $500,000 (20%)
    - Operations: $400,000 (16%)
    - Emergency Fund: $200,000 (8%)

    SECTION 2: DEBT MANAGEMENT
    The company maintains a conservative debt-to-equity ratio of 0.3:1. Our current debt obligations include:
    - Long-term loans: $1.2 million at 4.5% interest
    - Credit lines: $500,000 at 3.8% interest
    - Equipment financing: $300,000 at 5.2% interest

    Debt management policies:
    1. Maximum debt-to-equity ratio: 0.5:1
    2. Minimum credit rating requirement: BBB
    3. Regular debt review: Quarterly
    4. Early repayment incentives available

    SECTION 3: INFRASTRUCTURE INVESTMENT
    Infrastructure investments for 2025 focus on:
    - Cloud computing upgrades: $200,000
    - Security enhancements: $150,000
    - Network improvements: $100,000
    - Office renovations: $100,000
    - Equipment replacement: $50,000

    SECTION 4: COMPLIANCE REQUIREMENTS
    All financial activities must comply with:
    - Sarbanes-Oxley Act (SOX)
    - Generally Accepted Accounting Principles (GAAP)
    - Internal Revenue Service (IRS) regulations
    - State and local tax laws

    Non-compliance consequences:
    - First violation: Written warning
    - Second violation: Suspension
    - Third violation: Termination
    - Legal violations: Immediate termination and legal action

    SECTION 5: REPORTING REQUIREMENTS
    Financial reports must be submitted:
    - Monthly: Basic financial statements
    - Quarterly: Detailed analysis and projections
    - Annually: Comprehensive audit and review

    SECTION 6: APPROVAL PROCESS
    Financial decisions require approval based on amount:
    - $0 - $10,000: Department Manager
    - $10,001 - $50,000: Finance Director
    - $50,001 - $200,000: CFO
    - $200,001+: Board of Directors

    SECTION 7: RISK MANAGEMENT
    Risk assessment is conducted annually with:
    - Market risk analysis
    - Credit risk evaluation
    - Operational risk assessment
    - Compliance risk review

    SECTION 8: SUSTAINABILITY INITIATIVES
    Environmental and social responsibility investments:
    - Green energy projects: $100,000
    - Community programs: $75,000
    - Employee wellness: $50,000
    - Carbon offset programs: $25,000
    """
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        return f.name

def run_demo():
    """Run the chatbot demonstration."""
    print("=" * 60)
    print("FINANCIAL POLICY CHATBOT DEMO")
    print("=" * 60)
    print()
    
    # Create sample document
    print("📄 Creating sample financial policy document...")
    sample_file = create_sample_policy()
    print(f"✅ Sample document created: {sample_file}")
    print()
    
    try:
        # Initialize chatbot
        print("🤖 Initializing Financial Policy Chatbot...")
        chatbot = FinancialPolicyChatbot()
        print("✅ Chatbot initialized successfully!")
        print()
        
        # Load and process document
        print("📖 Loading and processing document...")
        documents = chatbot.load_document(sample_file)
        chatbot.process_documents(documents)
        print("✅ Document processed successfully!")
        print()
        
        # Show document summary
        summary = chatbot.get_document_summary()
        print("📊 Document Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        print()
        
        # Demo questions
        demo_questions = [
            "What is the total budget allocation for 2025?",
            "What is the debt-to-equity ratio?",
            "What are the consequences of non-compliance?",
            "How much is allocated for infrastructure investment?",
            "What is the approval process for a $25,000 expense?",
            "What sustainability initiatives are planned?",
            "What are the reporting requirements?",
            "What is the maximum debt-to-equity ratio allowed?"
        ]
        
        print("🧪 Running Demo Questions:")
        print("-" * 60)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n🔍 Question {i}: {question}")
            print("-" * 40)
            
            try:
                response = chatbot.ask_question(question)
                print(f"💡 Answer: {response['answer']}")
                
                if response['sources']:
                    print(f"\n📚 Sources ({len(response['sources'])} found):")
                    for j, source in enumerate(response['sources'], 1):
                        print(f"   {j}. {source['metadata']}")
                        print(f"      Content: {source['content']}")
                else:
                    print("\n📚 No specific sources found.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
        
        # Test conversation memory
        print("\n🧠 Testing Conversation Memory:")
        print("-" * 60)
        
        # First question
        print("\n🤔 Question: What is the budget for Research and Development?")
        response1 = chatbot.ask_question("What is the budget for Research and Development?")
        print(f"💡 Answer: {response1['answer']}")
        
        # Follow-up question (should remember context)
        print("\n🤔 Follow-up: What about Marketing?")
        response2 = chatbot.ask_question("What about Marketing?")
        print(f"💡 Answer: {response2['answer']}")
        
        # Another follow-up
        print("\n🤔 Follow-up: And Operations?")
        response3 = chatbot.ask_question("And Operations?")
        print(f"💡 Answer: {response3['answer']}")
        
        # Show chat history
        print("\n📚 Chat History:")
        print("-" * 60)
        history = chatbot.get_chat_history()
        for i, chat in enumerate(history, 1):
            print(f"\n💬 Conversation {i}:")
            print(f"   Question: {chat['question']}")
            print(f"   Answer: {chat['answer'][:100]}...")
            print(f"   Sources: {len(chat['sources'])} found")
        
        # Test document search
        print("\n🔍 Testing Document Search:")
        print("-" * 60)
        
        search_terms = ["budget", "debt", "infrastructure", "compliance"]
        for term in search_terms:
            print(f"\n🔎 Searching for: '{term}'")
            results = chatbot.search_documents(term, k=2)
            if results:
                print(f"   Found {len(results)} relevant sections:")
                for j, doc in enumerate(results, 1):
                    print(f"   {j}. {doc.metadata}")
                    print(f"      Preview: {doc.page_content[:100]}...")
            else:
                print(f"   No results found for '{term}'")
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe chatbot successfully demonstrated:")
        print("✅ Document loading and processing")
        print("✅ Question answering with RAG")
        print("✅ Source tracking and metadata")
        print("✅ Conversation memory and context")
        print("✅ Document search capabilities")
        print("✅ Error handling and fallbacks")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(sample_file)
            print(f"\n🧹 Cleaned up temporary file: {sample_file}")
        except:
            pass

if __name__ == "__main__":
    run_demo()