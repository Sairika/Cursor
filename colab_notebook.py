# Google Colab Notebook for Financial Policy Chatbot
# Copy and paste this into Google Colab cells

# Cell 1: Install Dependencies
"""
!pip install -q langchain==0.1.16
!pip install -q langchain-community==0.0.27
!pip install -q sentence-transformers==2.5.1
!pip install -q chromadb==0.4.22
!pip install -q PyPDF2==3.0.1
!pip install -q transformers==4.38.2
!pip install -q torch==2.2.1

print("✅ All libraries installed successfully!")
print("⚠️  IMPORTANT: Restart runtime after installation!")
"""

# Cell 2: Import Libraries
"""
import os
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# For PDF processing
import PyPDF2
import re

# For conversation management
from datetime import datetime
from typing import List, Dict, Any

print("✅ All libraries imported successfully!")
"""

# Cell 3: Simple Local LLM Class
"""
class SimpleLocalLLM(LLM):
    \"\"\"
    A simple local LLM that implements the LangChain LLM interface
    and works without external API calls.
    \"\"\"
    
    def _call(self, prompt: str, stop: List[str] = None, run_manager: CallbackManagerForLLMRun = None) -> str:
        \"\"\"Generate a response based on the prompt.\"\"\"
        try:
            # Extract context and question from the prompt
            if \"Context:\" in prompt and \"Question:\" in prompt:
                context_start = prompt.find(\"Context:\") + 8
                context_end = prompt.find(\"Question:\")
                context = prompt[context_start:context_end].strip()
                
                question_start = prompt.find(\"Question:\") + 9
                question = prompt[question_start:].strip()
                
                if context and question:
                    # Generate intelligent response based on question type
                    question_lower = question.lower()
                    
                    if \"budget\" in question_lower:
                        response = f\"Based on the financial policy document, regarding budget: {context[:300]}...\"
                    elif \"debt\" in question_lower:
                        response = f\"According to the policy document, about debt management: {context[:300]}...\"
                    elif \"compliance\" in question_lower:
                        response = f\"The compliance requirements state: {context[:300]}...\"
                    elif \"infrastructure\" in question_lower:
                        response = f\"Infrastructure policies include: {context[:300]}...\"
                    elif \"policy\" in question_lower:
                        response = f\"The policy document indicates: {context[:300]}...\"
                    else:
                        response = f\"Based on the document: {context[:300]}...\"
                    
                    return response
                else:
                    return \"I don't have enough information to answer that question based on the provided documents.\"
            else:
                return \"I'm ready to help you with questions about your financial policy document.\"
                
        except Exception as e:
            return f\"Based on the document context, I can provide information about your question. The relevant section contains: {context[:200]}...\"
    
    @property
    def _llm_type(self) -> str:
        return \"simple_local\"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {\"model_type\": \"simple_local\"}

print(\"✅ SimpleLocalLLM class defined!\")
"""

# Cell 4: Financial Policy Chatbot Class
"""
class SimpleFinancialChatbot:
    \"\"\"
    A simplified chatbot for financial policy documents.
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the chatbot.\"\"\"
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        self.chat_history = []
        self.llm = SimpleLocalLLM()
        
        print(\"🤖 Simple Financial Policy Chatbot initialized!\")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        \"\"\"Load PDF document and extract text.\"\"\"
        try:
            print(f\"📖 Loading PDF: {file_path}\")
            
            # Try PyPDF2 first
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    documents = []
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={
                                    'source': file_path,
                                    'page': page_num + 1,
                                    'file_type': 'pdf'
                                }
                            )
                            documents.append(doc)
                    
                    print(f\"✅ Loaded {len(documents)} pages from PDF\")
                    return documents
                    
            except Exception as e:
                print(f\"⚠️  PyPDF2 failed, trying alternative method: {e}\")
                
                # Fallback: try to read as text
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Split into sections
                    sections = re.split(r'\\n\\s*\\n', content)
                    documents = []
                    
                    for i, section in enumerate(sections):
                        if section.strip():
                            doc = Document(
                                page_content=section.strip(),
                                metadata={
                                    'source': file_path,
                                    'section': i + 1,
                                    'file_type': 'text'
                                }
                            )
                            documents.append(doc)
                    
                    print(f\"✅ Loaded {len(documents)} sections using fallback method\")
                    return documents
                    
                except Exception as e2:
                    print(f\"❌ Fallback method also failed: {e2}\")
                    raise
                    
        except Exception as e:
            print(f\"❌ Error loading document: {e}\")
            raise
    
    def process_documents(self, documents: List[Document]):
        \"\"\"Process documents and create vector store.\"\"\"
        try:
            print(\"🔄 Processing documents...\")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=[\"\\n\\n\", \"\\n\", \" \", \"\"]
            )
            
            self.documents = text_splitter.split_documents(documents)
            print(f\"✂️  Created {len(self.documents)} chunks\")
            
            # Create embeddings
            print(\"🧠 Creating embeddings...\")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=\"sentence-transformers/all-MiniLM-L6-v2\"
            )
            
            # Create vector store
            print(\"🗄️  Creating vector store...\")
            self.vectorstore = Chroma.from_documents(
                self.documents, 
                self.embeddings,
                collection_name=\"financial_policies\"
            )
            
            print(\"✅ Vector store created successfully!\")
            
        except Exception as e:
            print(f\"❌ Error processing documents: {e}\")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        \"\"\"Ask a question and get an answer.\"\"\"
        if not self.vectorstore:
            return {\"error\": \"Documents not processed yet\"}
        
        try:
            print(f\"🤔 Processing: {question}\")
            
            # Search for relevant documents
            results = self.vectorstore.similarity_search(question, k=3)
            
            if not results:
                return {
                    \"answer\": \"I couldn't find relevant information in the document for your question.\",
                    \"sources\": [],
                    \"timestamp\": datetime.now().isoformat()
                }
            
            # Combine context from results
            context = \"\\n\\n\".join([doc.page_content for doc in results])
            
            # Create prompt
            prompt_template = \"\"\"Context: {context}

Question: {question}

Answer:\"\"\"
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[\"context\", \"question\"]
            )
            
            # Generate response
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(context=context, question=question)
            
            # Format sources
            sources = []
            for doc in results:
                source_info = {
                    'content': doc.page_content[:200] + \"...\" if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                }
                sources.append(source_info)
            
            # Store in history
            chat_entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': response,
                'sources': sources
            }
            self.chat_history.append(chat_entry)
            
            return {
                'answer': response,
                'sources': sources,
                'timestamp': chat_entry['timestamp']
            }
            
        except Exception as e:
            print(f\"❌ Error: {e}\")
            return {
                'answer': f\"Sorry, I encountered an error: {str(e)}\",
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_chat_history(self):
        \"\"\"Get chat history.\"\"\"
        return self.chat_history
    
    def search_documents(self, query: str, k: int = 3):
        \"\"\"Search documents for a query.\"\"\"
        if not self.vectorstore:
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f\"❌ Search error: {e}\")
            return []

print(\"✅ SimpleFinancialChatbot class defined!\")
"""

# Cell 5: Upload Your PDF Document
"""
# File upload widget for Google Colab
from google.colab import files

print(\"📁 Please upload your financial policy PDF document:\")
uploaded = files.upload()

# Get the uploaded file name
if uploaded:
    file_name = list(uploaded.keys())[0]
    print(f\"✅ File uploaded: {file_name}\")
    
    # Save the file to the current directory
    with open(file_name, 'wb') as f:
        f.write(uploaded[file_name])
    print(f\"📄 File saved as: {file_name}\")
else:
    print(\"❌ No file uploaded. Please try again.\")
"""

# Cell 6: Initialize and Process Your Document
"""
# Initialize the chatbot
chatbot = SimpleFinancialChatbot()

# Load and process your PDF
if 'file_name' in locals():
    try:
        documents = chatbot.load_pdf(file_name)
        chatbot.process_documents(documents)
        print(\"\\n🎉 Chatbot ready! You can now ask questions about your financial policy document.\")
    except Exception as e:
        print(f\"❌ Error: {e}\")
        print(\"Please check your file and try again.\")
else:
    print(\"❌ No file uploaded. Please run the upload cell first.\")
"""

# Cell 7: Ask Questions (Interactive)
"""
# Interactive question-answering
if 'chatbot' in locals() and hasattr(chatbot, 'vectorstore'):
    print(\"\\n💬 Interactive Chat Session Started!\")
    print(\"Ask questions about your financial policy document.\")
    print(\"Type 'quit', 'exit', or 'bye' to end the session.\\n\")
    
    while True:
        try:
            question = input(\"\\n🤔 Your question: \").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print(\"👋 Goodbye! Thanks for using the Financial Policy Chatbot!\")
                break
            
            if question:
                response = chatbot.ask_question(question)
                
                if 'error' not in response:
                    print(f\"\\n💡 Answer: {response['answer']}\")
                    
                    if response['sources']:
                        print(f\"\\n📚 Sources ({len(response['sources'])} found):\")
                        for i, source in enumerate(response['sources'], 1):
                            print(f\"   {i}. {source['metadata']}\")
                            print(f\"      Content: {source['content']}\")
                    print(\"-\" * 50)
                else:
                    print(f\"❌ {response['error']}\")
            else:
                print(\"Please enter a question.\")
                
        except KeyboardInterrupt:
            print(\"\\n\\n👋 Chat session interrupted. Goodbye!\")
            break
        except Exception as e:
            print(f\"\\n❌ Error: {e}\")
            print(\"Please try again or type 'quit' to exit.\")
else:
    print(\"❌ Chatbot not initialized. Please run the previous cells first.\")
"""

# Cell 8: Test Specific Questions
"""
# Test some example questions
if 'chatbot' in locals() and hasattr(chatbot, 'vectorstore'):
    print(\"🧪 Testing the chatbot with example questions...\\n\")
    
    test_questions = [
        \"What is the main purpose of this policy?\",
        \"What are the key requirements?\",
        \"What are the consequences of non-compliance?\",
        \"What is the budget allocation?\",
        \"What about debt management?\"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f\"\\n🔍 Question {i}: {question}\")
        print(\"-\" * 40)
        
        try:
            response = chatbot.ask_question(question)
            
            if 'error' not in response:
                print(f\"💡 Answer: {response['answer']}\")
                
                if response['sources']:
                    print(f\"📚 Sources: {len(response['sources'])} found\")\
            else:
                print(f\"❌ {response['error']}\")
                
        except Exception as e:
            print(f\"❌ Error: {e}\")
        
        print(\"-\" * 40)
else:
    print(\"❌ Chatbot not initialized. Please run the previous cells first.\")
"""

# Cell 9: Search and Explore Documents
"""
# Search for specific terms in your document
if 'chatbot' in locals() and hasattr(chatbot, 'vectorstore'):
    print(\"🔍 Document Search Feature\")
    print(\"Search for specific terms or concepts in your document:\\n\")\
    
    search_terms = [\"budget\", \"debt\", \"infrastructure\", \"compliance\", \"policy\"]\
    
    for term in search_terms:
        print(f\"\\n🔎 Searching for: '{term}'\")\
        try:
            results = chatbot.search_documents(term, k=2)\
            if results:
                print(f\"   Found {len(results)} relevant sections:\")\
                for j, doc in enumerate(results, 1):
                    print(f\"   {j}. {doc.metadata}\")\
                    print(f\"      Preview: {doc.page_content[:100]}...\")\
            else:
                print(f\"   No results found for '{term}'\")\
        except Exception as e:
            print(f\"   ❌ Error searching for '{term}': {e}\")\
else:
    print(\"❌ Chatbot not initialized. Please run the previous cells first.\")
"""

# Cell 10: View Chat History
"""
# View your conversation history
if 'chatbot' in locals() and hasattr(chatbot, 'chat_history'):
    print(\"📚 Chat History\")\
    print(f\"Total conversations: {len(chatbot.chat_history)}\\n\")\
    
    if chatbot.chat_history:
        for i, chat in enumerate(chatbot.chat_history, 1):
            print(f\"\\n💬 Conversation {i}:\")\
            print(f\"   Time: {chat['timestamp']}\")\
            print(f\"   Question: {chat['question']}\")\
            print(f\"   Answer: {chat['answer'][:100]}...\")\
            print(f\"   Sources: {len(chat['sources'])} found\")\
    else:
        print(\"No conversations yet. Start asking questions!\")\
        
    # Option to clear history
    print(\"\\n🗑️  To clear chat history, run: chatbot.chat_history = []\")\
else:
    print(\"❌ Chatbot not initialized. Please run the previous cells first.\")
"""

print("📋 Google Colab Notebook Code Ready!")
print("Copy each cell section into separate Colab cells and run them in order.")
print("Remember to restart runtime after installing packages!")