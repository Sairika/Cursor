#!/usr/bin/env python3
"""
Improved Financial Policy Chatbot
Enhanced version with better responses, cleaner output, and improved conversation memory
using the working flan-t5 model.

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
from langchain.llms import HuggingFacePipeline

# For PDF processing
import PyPDF2

# For conversation management
from datetime import datetime
from typing import List, Dict, Any
import re

class ImprovedFinancialPolicyChatbot:
    """
    An improved chatbot for financial policy documents with better responses and memory.
    """
    
    def __init__(self):
        """Initialize the chatbot."""
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.documents = []
        self.chat_history = []
        self.llm = None
        
        print("🤖 Improved Financial Policy Chatbot initialized!")
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load PDF or text document with better error handling."""
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
        """Load PDF document with improved text extraction."""
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
        """Clean and normalize extracted text."""
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
        """Process documents with improved chunking strategy."""
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
        """Create improved QA chain with better prompting."""
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
        """Get improved LLM with better parameters."""
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
        """Create improved fallback LLM."""
        class ImprovedFallbackLLM:
            def __call__(self, prompt):
                if "Context:" in prompt and "Question:" in prompt:
                    context_start = prompt.find("Context:") + 8
                    context_end = prompt.find("Question:")
                    context = prompt[context_start:context_end].strip()
                    
                    question_start = prompt.find("Question:") + 9
                    question = prompt[question_start:].strip()
                    
                    if context and question:
                        # Generate better structured responses
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
        """Ask a question with improved response handling."""
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
        """Improve the quality of the generated answer."""
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
        """Extract the most relevant content for the source."""
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
        """Get improved chat history."""
        return self.chat_history
    
    def search_documents(self, query: str, k: int = 3):
        """Search documents with better results."""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded documents."""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        try:
            total_chunks = len(self.documents)
            total_characters = sum(len(doc.page_content) for doc in self.documents)
            
            # Get unique sources
            sources = set()
            for doc in self.documents:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            return {
                'total_chunks': total_chunks,
                'total_characters': total_characters,
                'sources': list(sources),
                'average_chunk_size': total_characters / total_chunks if total_chunks > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return {"error": str(e)}

def main():
    """Main function."""
    print("=" * 60)
    print("IMPROVED FINANCIAL POLICY CHATBOT")
    print("=" * 60)
    print()
    
    # Initialize chatbot
    chatbot = ImprovedFinancialPolicyChatbot()
    
    # Get file path
    file_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Load and process
        documents = chatbot.load_document(file_path)
        chatbot.process_documents(documents)
        
        print("\n🎉 Improved chatbot ready! Ask questions about your document.")
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
                            print(f"   {i}. Page {source['metadata'].get('page', 'N/A')}")
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