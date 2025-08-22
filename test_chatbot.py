#!/usr/bin/env python3
"""
Test Script for Financial Policy Chatbot
Join Venture AI (JVAI) - AI Developer Assessment

This script tests the basic functionality of the chatbot without requiring
the full IBM Watson setup, making it useful for validation.
"""

import os
import sys
from typing import List, Dict, Any

def test_document_loading():
    """Test if the financial policy document can be loaded"""
    print("🧪 Testing document loading...")
    
    try:
        if not os.path.exists("financial_policy_document.txt"):
            print("❌ Financial policy document not found!")
            return False
        
        with open("financial_policy_document.txt", 'r') as f:
            content = f.read()
        
        if len(content) > 0:
            print(f"✅ Document loaded successfully! Length: {len(content)} characters")
            return True
        else:
            print("❌ Document is empty!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return False

def test_chatbot_class():
    """Test if the chatbot class can be imported and instantiated"""
    print("\n🧪 Testing chatbot class...")
    
    try:
        # Try to import the chatbot class
        from financial_chatbot import FinancialPolicyChatbot, ChatbotConfig
        
        print("✅ Chatbot class imported successfully!")
        
        # Test configuration
        config = ChatbotConfig()
        print(f"✅ Configuration created: chunk_size={config.chunk_size}, model_id={config.model_id}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This is expected if IBM Watson packages are not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing chatbot class: {e}")
        return False

def test_dependencies():
    """Test if required dependencies can be imported"""
    print("\n🧪 Testing dependencies...")
    
    dependencies = [
        ("langchain", "langchain"),
        ("langchain-ibm", "langchain_ibm"),
        ("ibm-watsonx-ai", "ibm_watsonx_ai"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("chromadb", "chromadb")
    ]
    
    all_available = True
    
    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {package_name} - Available")
        except ImportError:
            print(f"❌ {package_name} - Not available")
            all_available = False
    
    return all_available

def test_document_structure():
    """Test the structure of the financial policy document"""
    print("\n🧪 Testing document structure...")
    
    try:
        with open("financial_policy_document.txt", 'r') as f:
            content = f.read()
        
        # Check for key sections
        key_sections = [
            "BUDGET ALLOCATION",
            "DEBT MANAGEMENT",
            "INFRASTRUCTURE INVESTMENT",
            "EXPENSE POLICIES",
            "FINANCIAL REPORTING",
            "COMPLIANCE AND AUDITING",
            "INVESTMENT POLICY",
            "EMERGENCY FUNDS",
            "PERFORMANCE METRICS"
        ]
        
        found_sections = []
        for section in key_sections:
            if section in content:
                found_sections.append(section)
                print(f"✅ Found section: {section}")
            else:
                print(f"❌ Missing section: {section}")
        
        if len(found_sections) >= 7:  # At least 7 out of 9 sections
            print(f"✅ Document structure looks good! Found {len(found_sections)}/9 key sections")
            return True
        else:
            print(f"❌ Document structure incomplete. Found {len(found_sections)}/9 sections")
            return False
            
    except Exception as e:
        print(f"❌ Error testing document structure: {e}")
        return False

def test_sample_questions():
    """Test if the document contains answers to sample questions"""
    print("\n🧪 Testing sample questions...")
    
    try:
        with open("financial_policy_document.txt", 'r') as f:
            content = f.read()
        
        sample_questions = [
            ("What is the total annual budget?", "$2,500,000"),
            ("What are the debt limits?", "$500,000"),
            ("What is the travel policy for domestic trips?", "$500 per day"),
            ("How often are budget reviews conducted?", "15th of each month"),
            ("What are the emergency fund requirements?", "$500,000")
        ]
        
        correct_answers = 0
        
        for question, expected_answer in sample_questions:
            if expected_answer in content:
                print(f"✅ Found answer for: {question}")
                correct_answers += 1
            else:
                print(f"❌ Missing answer for: {question}")
        
        if correct_answers >= 4:  # At least 4 out of 5 questions
            print(f"✅ Sample questions test passed! {correct_answers}/5 answers found")
            return True
        else:
            print(f"❌ Sample questions test failed. {correct_answers}/5 answers found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sample questions: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting Financial Policy Chatbot Tests...")
    print("=" * 60)
    
    tests = [
        ("Document Loading", test_document_loading),
        ("Chatbot Class", test_chatbot_class),
        ("Dependencies", test_dependencies),
        ("Document Structure", test_document_structure),
        ("Sample Questions", test_sample_questions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The chatbot is ready to use.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Some dependencies may be missing.")
    else:
        print("❌ Many tests failed. Please check the setup.")
    
    return passed, total

if __name__ == "__main__":
    try:
        passed, total = run_all_tests()
        
        if passed == total:
            print("\n🚀 Ready to run the full chatbot!")
            print("To start the chatbot, run: python financial_chatbot.py")
        else:
            print(f"\n⚠️  {total - passed} tests failed. Please address the issues above.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)