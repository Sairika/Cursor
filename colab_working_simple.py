# Google Colab Working Financial Policy Chatbot
# Copy and paste each section into separate Colab cells

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
!pip install -q langchain==0.1.16
!pip install -q langchain-community==0.0.27
!pip install -q sentence-transformers==2.5.1
!pip install -q chromadb==0.4.22
!pip install -q PyPDF2==3.0.1
!pip install -q transformers==4.38.2
!pip install -q torch==2.2.1
!pip install -q accelerate==0.27.2

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

print("✅ All libraries imported successfully!")
"""

# ============================================================================
# CELL 3: Working Chatbot Class
# ============================================================================
"""
class FinancialPolicyChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.documents = []
        self.chat_history = []
        print("🤖 Financial Policy Chatbot initialized!")
        
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
                        doc = Document(
                            page_content=text,
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
    
    def process_documents(self, documents: List[Document]):
        try:
            print("🔄 Processing documents...")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\\n\\n", "\\n", " ", ""]
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
            
            # Initialize memory
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
            # Custom prompt template
            prompt_template = \"\"\"You are a helpful AI assistant that answers questions about financial policy documents. 
            Use the following context to answer the question. If you cannot find the answer in the context, 
            say "I don't have enough information to answer that question based on the provided documents."
            
            Context: {context}
            
            Question: {question}
            
            Answer: Provide a clear, helpful answer based on the context. Include specific details when available.\"\"\"
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create conversational retrieval chain
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
            
            print("✅ QA chain created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating QA chain: {e}")
            raise
    
    def _get_llm(self):
        try:
            # Try flan-t5 first (most reliable on Colab)
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            import torch
            
            print("🔄 Loading flan-t5 model...")
            
            # Use smaller model for Colab memory
            model_name = "google/flan-t5-base"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Create LangChain wrapper
            from transformers import pipeline
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            print("✅ flan-t5 model loaded successfully!")
            return llm
            
        except Exception as e:
            print(f"⚠️  flan-t5 failed: {e}")
            
            try:
                # Fallback to llama-2-7b-chat-hf
                print("🔄 Trying llama-2-7b-chat-hf...")
                
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                model_name = "meta-llama/Llama-2-7b-chat-hf"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                
                from transformers import pipeline
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                print("✅ llama-2 model loaded successfully!")
                return llm
                
            except Exception as e2:
                print(f"⚠️  llama-2 also failed: {e2}")
                print("🔄 Using fallback responses...")
                return self._create_fallback_llm()
    
    def _create_fallback_llm(self):
        class FallbackLLM:
            def __call__(self, prompt):
                if "Context:" in prompt and "Question:" in prompt:
                    context_start = prompt.find("Context:") + 8
                    context_end = prompt.find("Question:")
                    context = prompt[context_start:context_end].strip()
                    
                    if context:
                        return f"Based on the financial policy document: {context[:300]}..."
                    else:
                        return "I don't have enough information to answer that question."
                else:
                    return "I'm ready to help with your financial policy questions."
        
        return FallbackLLM()
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"error": "Documents not processed yet"}
        
        try:
            print(f"🤔 Processing: {question}")
            
            # Get response
            response = self.qa_chain({"question": question})
            
            answer = response.get("answer", "No answer generated")
            source_docs = response.get("source_documents", [])
            
            # Format sources
            sources = []
            for doc in source_docs:
                source_info = {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                }
                sources.append(source_info)
            
            # Store in history
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
            print(f"❌ Error: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
    
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

print("✅ FinancialPolicyChatbot class defined successfully!")
"""

# ============================================================================
# CELL 4: Upload Your PDF Document
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
# Initialize the chatbot
chatbot = FinancialPolicyChatbot()

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
# CELL 6: Test Your Chatbot
# ============================================================================
"""
# Test some example questions
if 'chatbot' in locals() and hasattr(chatbot, 'qa_chain'):
    print("🧪 Testing the chatbot with example questions...\\n")
    
    test_questions = [
        "What is the main purpose of this policy?",
        "What are the key requirements?",
        "What are the consequences of non-compliance?",
        "What is the budget allocation?",
        "What about debt management?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n🔍 Question {i}: {question}")
        try:
            response = chatbot.ask_question(question)
            
            if 'error' not in response:
                print(f"💡 Answer: {response['answer']}")
                
                if response['sources']:
                    print(f"📚 Sources: {len(response['sources'])} found")
                    for j, source in enumerate(response['sources'], 1):
                        print(f"   Source {j}: {source['metadata']}")
            else:
                print(f"❌ {response['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

# ============================================================================
# CELL 7: Interactive Chat Session
# ============================================================================
"""
# Interactive chat session
def interactive_chat():
    if 'chatbot' not in locals() or not hasattr(chatbot, 'qa_chain'):
        print("❌ Chatbot not initialized. Please run the previous cells first.")
        return
    
    print("\\n💬 Interactive Chat Session Started!")
    print("Ask questions about your financial policy document.")
    print("Type 'quit', 'exit', or 'bye' to end the session.\\n")
    
    while True:
        try:
            question = input("\\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for using the Financial Policy Chatbot!")
                break
            
            if question:
                response = chatbot.ask_question(question)
                
                if 'error' not in response:
                    print(f"\\n💡 Answer: {response['answer']}")
                    
                    if response['sources']:
                        print(f"\\n📚 Sources ({len(response['sources'])} found):")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"   {i}. {source['metadata']}")
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
# CELL 8: Search Documents
# ============================================================================
"""
# Search for specific terms in your document
if 'chatbot' in locals() and hasattr(chatbot, 'vectorstore'):
    print("🔍 Document Search Feature")
    print("Search for specific terms or concepts in your document:\\n")
    
    search_terms = ["budget", "debt", "infrastructure", "compliance", "policy"]
    
    for term in search_terms:
        print(f"\\n🔎 Searching for: '{term}'")
        try:
            results = chatbot.search_documents(term, k=2)
            if results:
                print(f"   Found {len(results)} relevant sections:")
                for j, doc in enumerate(results, 1):
                    print(f"   {j}. {doc.metadata}")
                    print(f"      Preview: {doc.page_content[:100]}...")
            else:
                print(f"   No results found for '{term}'")
        except Exception as e:
            print(f"   ❌ Error searching for '{term}': {e}")
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

# ============================================================================
# CELL 9: View Chat History
# ============================================================================
"""
# View your conversation history
if 'chatbot' in locals() and hasattr(chatbot, 'chat_history'):
    print("📚 Chat History")
    print(f"Total conversations: {len(chatbot.chat_history)}\\n")
    
    if chatbot.chat_history:
        for i, chat in enumerate(chatbot.chat_history, 1):
            print(f"\\n💬 Conversation {i}:")
            print(f"   Time: {chat['timestamp']}")
            print(f"   Question: {chat['question']}")
            print(f"   Answer: {chat['answer'][:100]}...")
            print(f"   Sources: {len(chat['sources'])} found")
    else:
        print("No conversations yet. Start asking questions!")
        
    # Option to clear history
    print("\\n🗑️  To clear chat history, run: chatbot.chat_history = []")
else:
    print("❌ Chatbot not initialized. Please run the previous cells first.")
"""

print("📋 Google Colab Working Chatbot Code Ready!")
print("Copy each cell section into separate Colab cells and run them in order.")
print("Remember to restart runtime after installing packages!")
print("This version uses flan-t5 and llama models that actually work on Colab!")