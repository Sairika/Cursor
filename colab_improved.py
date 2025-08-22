# Improved Google Colab Financial Policy Chatbot
# Copy and paste each section into separate Colab cells for better responses

# ============================================================================
# CELL 1: Install Dependencies (Same as before)
# ============================================================================
"""
!pip install -q langchain
!pip install -q langchain-community
!pip install -q sentence-transformers
!pip install -q chromadb
!pip install -q PyPDF2
!pip install -q transformers
!pip install -q torch
!pip install -q accelerate

print("✅ All libraries installed successfully!")
print("⚠️  IMPORTANT: Restart runtime after installation!")
"""

# ============================================================================
# CELL 2: Import Libraries (AFTER RESTARTING RUNTIME)
# ============================================================================
"""
import os
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
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline

# For PDF processing
import PyPDF2

# For conversation management
from datetime import datetime
from typing import List, Dict, Any
import re

print("✅ All libraries imported successfully!")
"""

# ============================================================================
# CELL 3: Improved Chatbot Class
# ============================================================================
"""
class ImprovedFinancialPolicyChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.documents = []
        self.chat_history = []
        self.llm = None
        print("🤖 Improved Financial Policy Chatbot initialized!")
        
    def load_document(self, file_path: str) -> List[Document]:
        try:
            if file_path.lower().endswith('.pdf'):
                print(f"📖 Loading PDF: {file_path}")
                return self._load_pdf(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_path}")
        except Exception as e:
            print(f"❌ Error loading document: {e}")
            raise
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                documents = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        # Clean up the extracted text
                        cleaned_text = self._clean_text(text)
                        if cleaned_text:
                            doc = Document(
                                page_content=cleaned_text,
                                metadata={
                                    'source': file_path,
                                    'page': page_num + 1,
                                    'file_type': 'pdf'
                                }
                            )
                            documents.append(doc)
                
                print(f"✅ Loaded {len(documents)} pages from PDF")
                return documents
                
        except Exception as e:
            print(f"❌ PDF loading failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'(\w)\s*(\w)', r'\1 \2', text)  # Fix merged words
        
        # Remove page numbers and headers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*of\s*\d+', '', text)
        
        # Clean up bullet points
        text = re.sub(r'•\s*', '• ', text)
        text = re.sub(r'-\s*', '- ', text)
        
        return text.strip()
    
    def process_documents(self, documents: List[Document]):
        try:
            print("🔄 Processing documents...")
            
            # Improved text splitting with better separators
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for better context
                chunk_overlap=150,  # Reduced overlap
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len
            )
            
            self.documents = text_splitter.split_documents(documents)
            print(f"✂️  Created {len(self.documents)} chunks")
            
            # Create embeddings
            print("🧠 Creating embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store
            print("🗄️  Creating vector store...")
            self.vectorstore = Chroma.from_documents(
                self.documents, 
                self.embeddings,
                collection_name="financial_policies"
            )
            
            print("✅ Vector store created successfully!")
            
            # Initialize improved memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create QA chain
            self._create_qa_chain()
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            raise
    
    def _create_qa_chain(self):
        try:
            # Enhanced prompt template for better responses
            prompt_template = """You are a professional financial policy analyst assistant. Your task is to provide clear, accurate, and helpful answers about financial policy documents.

IMPORTANT INSTRUCTIONS:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain enough information, say "Based on the provided document, I cannot find sufficient information to answer this question completely."
3. Provide structured, easy-to-read responses
4. Include specific details and numbers when available
5. Use bullet points for lists when appropriate
6. Be concise but comprehensive

Context: {context}

Question: {question}

Answer: Provide a clear, structured response based on the context above."""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self._get_llm(),
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Get more context
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False
            )
            
            print("✅ QA chain created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating QA chain: {e}")
            raise
    
    def _get_llm(self):
        try:
            # Use flan-t5 with optimized parameters
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            from transformers import pipeline
            import torch
            
            print("🔄 Loading improved flan-t5 model...")
            
            # Use base model for reliability
            model_name = "google/flan-t5-base"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Create optimized pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=256,  # Shorter for cleaner responses
                temperature=0.3,  # Lower temperature for more focused answers
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            self.llm = llm
            print("✅ Improved flan-t5 model loaded successfully!")
            return llm
            
        except Exception as e:
            print(f"❌ flan-t5 failed: {e}")
            return self._create_fallback_llm()
    
    def _create_fallback_llm(self):
        class ImprovedFallbackLLM:
            def __call__(self, prompt):
                if "Context:" in prompt and "Question:" in prompt:
                    context_start = prompt.find("Context:") + 8
                    context_end = prompt.find("Question:")
                    context = prompt[context_start:context_end].strip()
                    
                    question_start = prompt.find("Question:") + 9
                    question = prompt[question_start:].strip()
                    
                    if context and question:
                        question_lower = question.lower()
                        
                        if "budget" in question_lower:
                            return f"Based on the financial policy document:\n\n• Budget Information: {context[:200]}...\n\nThis information is extracted from the policy document to answer your question about budget allocation and management."
                        
                        elif "debt" in question_lower:
                            return f"Based on the financial policy document:\n\n• Debt Management: {context[:200]}...\n\nThis information is extracted from the policy document to answer your question about debt management policies."
                        
                        elif "compliance" in question_lower:
                            return f"Based on the financial policy document:\n\n• Compliance Requirements: {context[:200]}...\n\nThis information is extracted from the policy document to answer your question about compliance."
                        
                        else:
                            return f"Based on the financial policy document:\n\n• Relevant Information: {context[:300]}...\n\nThis information is extracted from the policy document to answer your question."
                    else:
                        return "I don't have enough information to answer that question based on the provided documents."
                else:
                    return "I'm ready to help you with questions about your financial policy document. Please ask me anything about the policies, budget, debt, infrastructure, or compliance requirements."
        
        return ImprovedFallbackLLM()
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"error": "Documents not processed yet"}
        
        try:
            print(f"🤔 Processing: {question}")
            
            # Get response
            response = self.qa_chain({"question": question})
            
            answer = response.get("answer", "No answer generated")
            source_docs = response.get("source_documents", [])
            
            # Clean and improve the answer
            improved_answer = self._improve_answer(answer, question)
            
            # Format sources with better information
            sources = []
            for doc in source_docs:
                source_info = {
                    'content': self._extract_key_content(doc.page_content, question),
                    'metadata': doc.metadata
                }
                sources.append(source_info)
            
            # Store in history
            chat_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': improved_answer,
                'sources': sources
            }
            self.chat_history.append(chat_entry)
            
            return {
                'answer': improved_answer,
                'sources': sources,
                'timestamp': chat_entry['timestamp']
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _improve_answer(self, answer: str, question: str) -> str:
        # Clean up common issues
        answer = re.sub(r'\s+', ' ', answer)  # Remove extra whitespace
        answer = re.sub(r'(\w)-\s*(\w)', r'\1\2', answer)  # Fix hyphenated words
        
        # Add structure if missing
        if not answer.startswith(('Based on', 'According to', 'The policy')):
            answer = f"Based on the financial policy document: {answer}"
        
        # Ensure proper sentence endings
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer.strip()
    
    def _extract_key_content(self, content: str, question: str) -> str:
        # Find the most relevant part of the content based on the question
        question_words = question.lower().split()
        content_lower = content.lower()
        
        # Look for sentences containing question words
        sentences = content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_words if len(word) > 3):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:2]) + "..."
        else:
            return content[:200] + "..." if len(content) > 200 else content
    
    def get_chat_history(self):
        return self.chat_history
    
    def search_documents(self, query: str, k: int = 3):
        if not self.vectorstore:
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

print("✅ ImprovedFinancialPolicyChatbot class defined successfully!")
"""

# ============================================================================
# CELL 4: Upload Your PDF Document (Same as before)
# ============================================================================
"""
# File upload widget for Google Colab
from google.colab import files

print("📁 Please upload your financial policy PDF document:")
uploaded = files.upload()

# Get the uploaded file name
if uploaded:
    file_name = list(uploaded.keys())[0]
    print(f"✅ File uploaded: {file_name}")
    
    # Save the file to the current directory
    with open(file_name, 'wb') as f:
        f.write(uploaded[file_name])
    print(f"📄 File saved as: {file_name}")
else:
    print("❌ No file uploaded. Please try again.")
"""

# ============================================================================
# CELL 5: Initialize and Process Your Document
# ============================================================================
"""
# Initialize the improved chatbot
chatbot = ImprovedFinancialPolicyChatbot()

# Load the uploaded document
if 'file_name' in locals():
    try:
        documents = chatbot.load_document(file_name)
        print(f"✅ Successfully loaded {len(documents)} document(s)")
        
        # Process the documents
        chatbot.process_documents(documents)
        print("✅ Documents processed and ready for questions!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your file and try again.")
else:
    print("❌ No file uploaded. Please run the file upload cell first.")
"""

# ============================================================================
# CELL 6: Test Improved Chatbot
# ============================================================================
"""
# Test the improved chatbot with better questions
if 'chatbot' in locals() and hasattr(chatbot, 'qa_chain'):
    print("🧪 Testing the improved chatbot...\\n")
    
    test_questions = [
        "What is the main purpose of this financial policy?",
        "What are the key budget requirements and strategies?",
        "What are the debt management policies?",
        "What are the compliance requirements?",
        "What infrastructure investments are planned?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n🔍 Question {i}: {question}")
        print("-" * 50)
        
        try:
            response = chatbot.ask_question(question)
            
            if 'error' not in response:
                print(f"💡 Answer: {response['answer']}")
                
                if response['sources']:
                    print(f"\\n📚 Sources ({len(response['sources'])} found):")
                    for j, source in enumerate(response['sources'], 1):
                        print(f"   {j}. Page {source['metadata'].get('page', 'N/A')}")
                        print(f"      Content: {source['content']}")
            else:
                print(f"❌ {response['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

# ============================================================================
# CELL 7: Interactive Chat with Memory
# ============================================================================
"""
# Interactive chat session with improved memory
def interactive_chat():
    if 'chatbot' not in locals() or not hasattr(chatbot, 'qa_chain'):
        print("❌ Chatbot not initialized. Please run the previous cells first.")
        return
    
    print("\\n💬 Interactive Chat Session Started!")
    print("Ask questions about your financial policy document.")
    print("The chatbot will remember previous questions for context.")
    print("Type 'quit', 'exit', or 'bye' to end the session.\\n")
    
    while True:
        try:
            question = input("\\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for using the Improved Financial Policy Chatbot!")
                break
            
            if question:
                response = chatbot.ask_question(question)
                
                if 'error' not in response:
                    print(f"\\n💡 Answer: {response['answer']}")
                    
                    if response['sources']:
                        print(f"\\n📚 Sources ({len(response['sources'])} found):")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"   {i}. Page {source['metadata'].get('page', 'N/A')}")
                            print(f"      Content: {source['content']}")
                    print("-" * 50)
                else:
                    print(f"❌ {response['error']}")
            else:
                print("Please enter a question.")
                
        except KeyboardInterrupt:
            print("\\n\\n👋 Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

# Start interactive chat
interactive_chat()
"""

# ============================================================================
# CELL 8: Test Conversation Memory
# ============================================================================
"""
# Test conversation memory with follow-up questions
if 'chatbot' in locals() and hasattr(chatbot, 'qa_chain'):
    print("🧠 Testing Conversation Memory...\\n")
    
    # First question
    print("🤔 Question 1: What is the main purpose of this policy?")
    response1 = chatbot.ask_question("What is the main purpose of this policy?")
    print(f"💡 Answer: {response1['answer']}")
    print("-" * 50)
    
    # Follow-up question (should remember context)
    print("\\n🤔 Follow-up: What about the budget requirements?")
    response2 = chatbot.ask_question("What about the budget requirements?")
    print(f"💡 Answer: {response2['answer']}")
    print("-" * 50)
    
    # Another follow-up
    print("\\n🤔 Follow-up: And debt management?")
    response3 = chatbot.ask_question("And debt management?")
    print(f"💡 Answer: {response3['answer']}")
    print("-" * 50)
    
    print("\\n✅ Memory test completed! The chatbot should have maintained context across questions.")
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

# ============================================================================
# CELL 9: Enhanced Document Search
# ============================================================================
"""
# Enhanced document search with better results
if 'chatbot' in locals() and hasattr(chatbot, 'vectorstore'):
    print("🔍 Enhanced Document Search")
    print("Search for specific terms with improved relevance:\\n")
    
    search_terms = ["budget", "debt", "infrastructure", "compliance", "policy", "financial"]
    
    for term in search_terms:
        print(f"\\n🔎 Searching for: '{term}'")
        try:
            results = chatbot.search_documents(term, k=3)
            if results:
                print(f"   Found {len(results)} relevant sections:")
                for j, doc in enumerate(results, 1):
                    print(f"   {j}. Page {doc.metadata.get('page', 'N/A')}")
                    print(f"      Preview: {doc.page_content[:150]}...")
            else:
                print(f"   No results found for '{term}'")
        except Exception as e:
            print(f"   ❌ Error searching for '{term}': {e}")
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

# ============================================================================
# CELL 10: View Enhanced Chat History
# ============================================================================
"""
# View enhanced chat history with better formatting
if 'chatbot' in locals() and hasattr(chatbot, 'chat_history'):
    print("📚 Enhanced Chat History")
    print(f"Total conversations: {len(chatbot.chat_history)}\\n")
    
    if chatbot.chat_history:
        for i, chat in enumerate(chatbot.chat_history, 1):
            print(f"\\n💬 Conversation {i}:")
            print(f"   Time: {chat['timestamp']}")
            print(f"   Question: {chat['question']}")
            print(f"   Answer: {chat['answer'][:150]}...")
            print(f"   Sources: {len(chat['sources'])} found")
            print(f"   Source Pages: {[s['metadata'].get('page', 'N/A') for s in chat['sources']]}")
    else:
        print("No conversations yet. Start asking questions!")
        
    # Option to clear history
    print("\\n🗑️  To clear chat history, run: chatbot.chat_history = []")
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

print("📋 Improved Google Colab Financial Policy Chatbot Ready!")
print("This version provides:")
print("✅ Better, cleaner responses")
print("✅ Improved conversation memory")
print("✅ Enhanced text cleaning")
print("✅ Better source tracking")
print("✅ Optimized flan-t5 parameters")
print("\nCopy each cell section into separate Colab cells and run them in order!")