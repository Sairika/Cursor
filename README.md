# AI-Powered Document Summarization Chatbot

A sophisticated chatbot that uses Retrieval-Augmented Generation (RAG), LangChain, and Large Language Models to summarize and answer questions about private documents.

## Features

- **Document Processing**: Automatically loads and processes various document formats
- **Smart Chunking**: Intelligently splits documents into manageable chunks for better processing
- **Vector Search**: Uses FAISS/ChromaDB for efficient document retrieval
- **Conversation Memory**: Maintains context across multiple questions
- **Multiple LLM Support**: Compatible with various open-source language models
- **Source Tracking**: Provides references to original document sections

## How to Run

### Option 1: Google Colab (Recommended for free usage)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `rag_chatbot.ipynb` file
3. Run the cells sequentially
4. Upload your document when prompted

### Option 2: Local Environment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python rag_chatbot.py
   ```

3. Or use the Streamlit interface:
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

```
├── rag_chatbot.py          # Main chatbot implementation
├── rag_chatbot.ipynb       # Jupyter notebook version
├── streamlit_app.py        # Web interface
├── document_processor.py   # Document loading and processing
├── vector_store.py         # Vector database operations
├── llm_interface.py        # Language model interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── sample_documents/      # Sample documents for testing
```

## Usage Examples

### Basic Document Processing
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
documents = processor.load_document("your_document.pdf")
chunks = processor.split_documents(documents)
```

### Chatbot Interaction
```python
from rag_chatbot import RAGChatbot

chatbot = RAGChatbot()
response = chatbot.ask("What are the main policies in this document?")
print(response)
```

## Supported Document Formats

- PDF files
- Text files (.txt)
- Word documents (.docx)
- Markdown files (.md)
- HTML files

## Model Options

The chatbot supports various open-source models:
- **Llama 2** (7B, 13B variants)
- **Mistral** (7B, Mixtral variants)
- **Falcon** (7B, 40B variants)
- **MPT** (7B, 30B variants)

## Key Components

### 1. Document Processing
- Automatic format detection
- Smart text extraction
- Intelligent chunking with overlap

### 2. Vector Database
- FAISS for fast similarity search
- ChromaDB for persistent storage
- Configurable embedding models

### 3. RAG Pipeline
- Context-aware retrieval
- Prompt engineering
- Response generation

### 4. Memory Management
- Conversation history tracking
- Context window management
- Session persistence

## Configuration

You can customize various parameters in `config.py`:
- Chunk size and overlap
- Model selection
- Vector database settings
- Memory configuration

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce model size or use CPU
2. **Document Loading Errors**: Check file format and permissions
3. **Slow Performance**: Optimize chunk size and use GPU if available

### Performance Tips

- Use smaller models for faster responses
- Optimize chunk size (500-1000 characters recommended)
- Enable GPU acceleration when available
- Use persistent vector storage for large documents

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.