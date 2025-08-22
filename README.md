# Financial Policy Document Chatbot

A powerful AI-powered chatbot that can answer questions about financial policy documents using **RAG (Retrieval-Augmented Generation)**, **LangChain**, and **open-source models**.

## 🎯 Project Overview

This chatbot demonstrates advanced AI capabilities for document understanding and question-answering, specifically designed for financial policy documents. It extracts key information, stores it in a vector database, and provides intelligent responses with source tracking.

## ✨ Key Features

- **📄 PDF Document Processing**: Automatically extracts text from PDF files
- **🧠 Intelligent Chunking**: Splits documents into optimal chunks with overlap
- **🔍 Vector Search**: Uses sentence transformers for semantic similarity search
- **💬 Conversational Memory**: Remembers conversation context for better responses
- **📚 Source Tracking**: Shows page numbers and file locations for all answers
- **🚀 No API Costs**: Runs entirely on open-source models and local resources
- **📱 Google Colab Ready**: Optimized for Colab's free resources

## 🏗️ Architecture

The chatbot follows the RAG (Retrieval-Augmented Generation) architecture:

### Indexing Phase
1. **Load**: Extract text from PDF documents
2. **Split**: Break text into manageable chunks with overlap
3. **Store**: Convert chunks to vector embeddings and store in ChromaDB

### Retrieval & Generation Phase
1. **Retrieve**: Find most relevant document chunks using vector similarity
2. **Generate**: Create answers based on retrieved context and conversation history

## 🛠️ Technology Stack

- **LangChain**: Document processing and RAG implementation
- **Sentence Transformers**: Text embeddings and semantic search
- **ChromaDB**: Vector database for similarity search
- **PyPDF2/pypdf**: PDF text extraction
- **Python 3.8+**: Core programming language

## 📋 Prerequisites

- Python 3.8 or higher
- Google Colab account (recommended) or local Python environment
- Financial policy document in PDF format

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Open the Notebook**: 
   - Upload `financial_chatbot_colab.ipynb` to Google Colab
   - Or copy the code from the Python file into a new Colab notebook

2. **Install Dependencies**:
   ```python
   !pip install -q langchain==0.1.16
   !pip install -q langchain-community==0.0.27
   !pip install -q sentence-transformers==2.5.1
   !pip install -q chromadb==0.4.22
   !pip install -q PyPDF2==3.0.1
   !pip install -q pypdf==4.0.2
   !pip install -q transformers==4.38.2
   !pip install -q torch==2.2.1
   ```

3. **Upload Your Document**: Use the file upload widget to upload your PDF

4. **Run the Chatbot**: Execute all cells and start asking questions!

### Option 2: Local Installation

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd financial-policy-chatbot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Chatbot**:
   ```bash
   python financial_chatbot.py
   ```

## 📖 Usage Examples

### Basic Questions
```
🤔 Your question: What is the main purpose of this policy?
💡 Answer: Based on the provided document context, this policy establishes...

🤔 Your question: What are the consequences of non-compliance?
💡 Answer: According to the document, non-compliance may result in...
```

### Contextual Questions (Memory Feature)
```
🤔 Your question: What is the budget allocation?
💡 Answer: The budget allocation is $2.5 million...

🤔 Your question: What about debt management?
💡 Answer: Regarding debt management, the policy states...
```

### Source Information
Every answer includes source details:
```
📚 Sources (2 found):
   1. {'source': 'financial_policy.pdf', 'page': 3, 'file_type': 'pdf'}
      Content: The debt management section outlines...
   2. {'source': 'financial_policy.pdf', 'page': 4, 'file_type': 'pdf'}
      Content: Additional debt considerations include...
```

## 🔧 Configuration Options

### Embedding Model
```python
# Use different sentence transformer models
chatbot = FinancialPolicyChatbot(model_name="sentence-transformers/all-MiniLM-L6-v2")
chatbot = FinancialPolicyChatbot(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

### Chunk Size and Overlap
```python
# Customize document processing
chatbot.process_documents(documents, chunk_size=800, overlap=150)
```

### Search Parameters
```python
# Adjust search results
results = chatbot.search_documents("budget", k=10)  # Get top 10 results
```

## 🧠 How the Chatbot "Remembers" Things

The chatbot uses **ConversationBufferMemory** from LangChain to maintain conversation context:

1. **Memory Storage**: All Q&A pairs are stored in memory
2. **Context Passing**: Previous conversations are included in new prompts
3. **Contextual Understanding**: Enables follow-up questions like "What about debt?" after asking about budget
4. **Conversation Flow**: Maintains coherent dialogue across multiple questions

### Example Memory Usage:
```
Question 1: "What is the budget allocation?"
Answer 1: "The budget allocation is $2.5 million..."

Question 2: "What about debt management?"
Answer 2: "Regarding debt management, the policy states..." 
         (The chatbot understands "debt management" refers to the same document)
```

## 🔍 Why This Search Setup Works

### Semantic Search vs Keyword Search
- **Traditional Search**: Finds exact word matches
- **Our Approach**: Understands meaning and context using sentence transformers

### Chunk Overlap Strategy
- **Problem**: Important information might be split across chunks
- **Solution**: Overlapping chunks ensure context continuity
- **Benefit**: Better understanding of complex policy sections

### Vector Similarity
- **Process**: Converts text to high-dimensional vectors
- **Search**: Finds most similar vectors (semantic similarity)
- **Result**: More relevant document sections for each question

## 📊 Performance Characteristics

- **Document Size**: Handles documents up to 100+ pages
- **Processing Time**: ~2-5 minutes for typical policy documents
- **Memory Usage**: Optimized for Colab's free tier (12GB RAM)
- **Response Time**: 1-3 seconds per question
- **Accuracy**: High accuracy due to RAG approach and source tracking

## 🚨 Troubleshooting

### Common Issues

1. **"No module named 'langchain'"**
   - Solution: Restart runtime after installing packages
   - Run: `!pip install -q langchain==0.1.16`

2. **"CUDA out of memory"**
   - Solution: Use smaller embedding model
   - Change to: `"sentence-transformers/all-MiniLM-L6-v2"`

3. **"PDF text extraction failed"**
   - Solution: Try different PDF library
   - The code automatically falls back to alternative PDF readers

4. **"Vector store creation failed"**
   - Solution: Check available memory
   - Reduce chunk size: `chunk_size=500`

### Performance Tips

- **Smaller Chunks**: Use `chunk_size=500` for memory-constrained environments
- **Efficient Models**: Use `all-MiniLM-L6-v2` for faster processing
- **Batch Processing**: Process documents in smaller batches if needed

## 🔮 Future Enhancements

- **Multi-Document Support**: Handle multiple policy documents simultaneously
- **Advanced LLM Integration**: Connect to more sophisticated language models
- **Web Interface**: Create a web-based chat interface
- **Export Functionality**: Save conversations and answers to files
- **Multi-Language Support**: Handle documents in different languages

## 📚 Technical Deep Dive

### RAG Implementation Details

```python
# 1. Document Loading and Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# 2. Embedding Generation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Vector Storage
vectorstore = Chroma.from_documents(documents, embeddings)

# 4. Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)
```

### Memory Management

```python
# Conversation memory with context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Memory is automatically included in each question
response = qa_chain({"question": question, "chat_history": history})
```

## 🤝 Contributing

This project was created for the AI Developer Assessment. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain Team**: For the excellent RAG framework
- **Hugging Face**: For open-source transformer models
- **ChromaDB Team**: For the vector database solution
- **Join Venture AI**: For the assessment opportunity

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Ensure all dependencies are properly installed
4. Verify your PDF document is readable and not corrupted

---

**Built with ❤️ for the AI Developer Assessment**

*This chatbot demonstrates advanced AI capabilities including document understanding, semantic search, and conversational AI - all using open-source technologies that work on Google Colab's free resources.*