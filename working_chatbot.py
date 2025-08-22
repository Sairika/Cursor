#!/usr/bin/env python3
"""
Working Financial Policy Chatbot
Uses flan-t5 and llama models that actually work on Google Colab
with proper RAG implementation and conversation memory.

Author: AI Developer Assessment
Date: 2025
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# For PDF processing
import PyPDF2

# For conversation management
from datetime import datetime
from typing import List, Dict, Any

class FinancialPolicyChatbot:
    """
    A working chatbot for financial policy documents using proper LLMs.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the chatbot.
        
        Args:
            model_name: HuggingFace model name (flan-t5 or llama variants)
        """
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.documents = []
        self.chat_history = []
        
        print(f"🤖 Financial Policy Chatbot initialized with {model_name}")
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load PDF or text document."""
        try:
            if file_path.lower().endswith('.pdf'):
                print(f"📖 Loading PDF: {file_path}")
                return self._load_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                print(f"📖 Loading text: {file_path}")
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_path}")
        except Exception as e:
            print(f"❌ Error loading document: {e}")
            raise
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document."""
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
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return [Document(
                page_content=text,
                metadata={
                    'source': file_path,
                    'file_type': 'text'
                }
            )]
            
        except Exception as e:
            print(f"❌ Text loading failed: {e}")
            raise
    
    def process_documents(self, documents: List[Document]):
        """Process documents and create vector store."""
        try:
            print("🔄 Processing documents...")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
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
        """Create the QA chain with proper LLM."""
        try:
            # Custom prompt template
            prompt_template = """You are a helpful AI assistant that answers questions about financial policy documents. 
            Use the following context to answer the question. If you cannot find the answer in the context, 
            say "I don't have enough information to answer that question based on the provided documents."
            
            Context: {context}
            
            Question: {question}
            
            Answer: Provide a clear, helpful answer based on the context. Include specific details when available."""
            
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
        """Get working LLM for Google Colab."""
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
            from langchain.llms import HuggingFacePipeline
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
                # Fallback to llama-2-7b-chat-hf (if available)
                print("🔄 Trying llama-2-7b-chat-hf...")
                
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                model_name = "meta-llama/Llama-2-7b-chat-hf"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                from langchain.llms import HuggingFacePipeline
                from transformers import pipeline
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                print("✅ llama-2 model loaded successfully!")
                return llm
                
            except Exception as e2:
                print(f"⚠️  llama-2 also failed: {e2}")
                
                # Final fallback: simple rule-based responses
                print("🔄 Using fallback rule-based responses...")
                return self._create_fallback_llm()
    
    def _create_fallback_llm(self):
        """Create fallback LLM when models fail."""
        class FallbackLLM:
            def __call__(self, prompt):
                if "Context:" in prompt and "Question:" in prompt:
                    context_start = prompt.find("Context:") + 8
                    context_end = prompt.find("Question:")
                    context = prompt[context_start:context_end].strip()
                    
                    question_start = prompt.find("Question:") + 9
                    question = prompt[question_start:].strip()
                    
                    if context and question:
                        return f"Based on the financial policy document: {context[:300]}..."
                    else:
                        return "I don't have enough information to answer that question."
                else:
                    return "I'm ready to help with your financial policy questions."
        
        return FallbackLLM()
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer."""
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
        """Get chat history."""
        return self.chat_history
    
    def search_documents(self, query: str, k: int = 3):
        """Search documents."""
        if not self.vectorstore:
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

def main():
    """Main function."""
    print("=" * 60)
    print("WORKING FINANCIAL POLICY CHATBOT")
    print("=" * 60)
    print()
    
    # Initialize chatbot
    chatbot = FinancialPolicyChatbot()
    
    # Get file path
    file_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Load and process
        documents = chatbot.load_document(file_path)
        chatbot.process_documents(documents)
        
        print("\n🎉 Chatbot ready! Ask questions about your document.")
        print("Type 'quit' to exit.\n")
        
        # Chat loop
        while True:
            question = input("🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if question:
                response = chatbot.ask_question(question)
                
                if 'error' not in response:
                    print(f"\n💡 Answer: {response['answer']}")
                    
                    if response['sources']:
                        print(f"\n📚 Sources ({len(response['sources'])} found):")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"   {i}. {source['metadata']}")
                            print(f"      Content: {source['content']}")
                    print("-" * 50)
                else:
                    print(f"❌ {response['error']}")
            else:
                print("Please enter a question.")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your file and try again.")

if __name__ == "__main__":
    main()