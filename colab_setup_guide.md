# Google Colab Setup Guide for Financial Policy Chatbot

This guide will walk you through setting up and running the Financial Policy Chatbot on Google Colab's free resources.

## 🚀 Quick Start (5 minutes)

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Create a new notebook or upload the provided notebook

### Step 2: Install Dependencies
Copy and paste this code into the first cell:

```python
# Install required libraries
!pip install -q langchain==0.1.16
!pip install -q langchain-community==0.0.27
!pip install -q sentence-transformers==2.5.1
!pip install -q chromadb==0.4.22
!pip install -q PyPDF2==3.0.1
!pip install -q pypdf==4.0.2
!pip install -q transformers==4.38.2
!pip install -q torch==2.2.1
!pip install -q accelerate==0.27.2

print("✅ All libraries installed successfully!")
```

**Important**: After installation, **restart the runtime** (Runtime → Restart runtime)

### Step 3: Import Libraries
In the next cell:

```python
# Import required libraries
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

# For PDF processing
import PyPDF2
from pypdf import PdfReader

# For embeddings and vector search
import chromadb
from sentence_transformers import SentenceTransformer

# For conversation management
import json
from datetime import datetime
from typing import List, Dict, Any

print("✅ All libraries imported successfully!")
```

### Step 4: Upload Your PDF
In the next cell:

```python
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
```

### Step 5: Copy the Chatbot Class
Copy the entire `FinancialPolicyChatbot` class from the `financial_chatbot.py` file into a new cell.

### Step 6: Initialize and Run
In the final cell:

```python
# Initialize the chatbot
chatbot = FinancialPolicyChatbot()

# Load and process documents
documents = chatbot.load_document(file_name)
chatbot.process_documents(documents)

print("✅ Chatbot ready! You can now ask questions about your financial policy document.")

# Test with a question
question = "What is the main purpose of this policy?"
response = chatbot.ask_question(question)
print(f"\nQuestion: {question}")
print(f"Answer: {response['answer']}")
```

## 🔧 Troubleshooting Common Issues

### Issue 1: "No module named 'langchain'"
**Solution**: Restart the runtime after installing packages
1. Go to Runtime → Restart runtime
2. Run the import cell again

### Issue 2: "CUDA out of memory"
**Solution**: Use smaller embedding model
```python
chatbot = FinancialPolicyChatbot(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Issue 3: "PDF text extraction failed"
**Solution**: The code automatically tries multiple PDF libraries
- PyPDF2 first
- Falls back to pypdf if needed

### Issue 4: "Vector store creation failed"
**Solution**: Reduce chunk size for memory-constrained environments
```python
chatbot.process_documents(documents, chunk_size=500, overlap=100)
```

## 📱 Colab-Specific Optimizations

### Memory Management
- **Free Tier**: 12GB RAM, 100GB disk
- **Recommended chunk size**: 500-1000 characters
- **Model choice**: Use `all-MiniLM-L6-v2` for efficiency

### Performance Tips
1. **Restart runtime** after installing packages
2. **Use smaller models** for faster processing
3. **Process documents in batches** if memory is limited
4. **Clear variables** when done: `del chatbot`

### GPU Acceleration
- Colab provides free GPU access (limited)
- To enable: Runtime → Change runtime type → GPU
- Not required for basic functionality

## 🎯 Example Usage

### Basic Questions
```python
# Ask about budget
response = chatbot.ask_question("What is the budget allocation?")
print(response['answer'])

# Ask about debt
response = chatbot.ask_question("What about debt management?")
print(response['answer'])
```

### Search Documents
```python
# Search for specific terms
results = chatbot.search_documents("budget", k=5)
for doc in results:
    print(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}...")
```

### Get Chat History
```python
# View conversation history
history = chatbot.get_chat_history()
for chat in history:
    print(f"Q: {chat['question']}")
    print(f"A: {chat['answer'][:100]}...")
```

## 🚨 Important Notes

1. **Runtime Disconnection**: Colab may disconnect after 12 hours of inactivity
2. **File Persistence**: Uploaded files are temporary - download results if needed
3. **Memory Limits**: Large documents may require chunk size reduction
4. **Model Downloads**: First run downloads embedding models (~100MB)

## 🔄 Complete Working Example

Here's a complete cell that you can copy-paste:

```python
# Complete working example
def setup_chatbot():
    # Initialize chatbot
    chatbot = FinancialPolicyChatbot()
    
    # Load and process documents
    documents = chatbot.load_document(file_name)
    chatbot.process_documents(documents)
    
    return chatbot

def ask_questions(chatbot):
    questions = [
        "What is the main purpose of this policy?",
        "What are the key requirements?",
        "What are the consequences of non-compliance?"
    ]
    
    for question in questions:
        print(f"\n🤔 Question: {question}")
        response = chatbot.ask_question(question)
        print(f"💡 Answer: {response['answer']}")
        print("-" * 50)

# Run the complete example
if 'file_name' in locals():
    chatbot = setup_chatbot()
    ask_questions(chatbot)
else:
    print("❌ Please upload a PDF file first!")
```

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ All libraries imported successfully
- ✅ File uploaded and saved
- ✅ Chatbot initialized successfully
- ✅ Documents processed and ready
- ✅ Questions answered with relevant information
- ✅ Source information displayed

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all cells are run in order
3. Restart runtime if needed
4. Check Colab's resource usage (Runtime → Manage sessions)

---

**Happy Chatbotting! 🤖✨**

*This guide ensures your Financial Policy Chatbot runs smoothly on Google Colab's free resources.*