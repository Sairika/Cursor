# 💰 FREE Financial Policy Chatbot
## Join Venture AI (JVAI) - AI Developer Assessment

**NO API KEYS NEEDED! NO EXTERNAL SERVICES! COMPLETELY FREE!**

## 🚀 What This Is

A simple AI chatbot that answers questions about financial policies using:
- **Vector search** (ChromaDB + HuggingFace)
- **Conversation memory** 
- **Local processing** - everything runs in Google Colab
- **Zero cost** - no external APIs or paid services

## 📋 Requirements

- Google Colab (free)
- Python packages (free)

## 🛠️ How to Use in Google Colab

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook

### Step 2: Copy the Code
Copy the code from `free_colab_chatbot.py` and paste it into separate cells in Colab.

### Step 3: Run Each Cell
Run the cells in order:

**Cell 1**: Install packages
```python
!pip install langchain chromadb sentence-transformers transformers torch
```

**Cell 2**: Import libraries
**Cell 3**: Create financial policy document  
**Cell 4**: Create chatbot class
**Cell 5**: Initialize chatbot
**Cell 6**: Test chatbot
**Cell 7**: Interactive chat function
**Cell 8**: Start chatting

### Step 4: Start Chatting
Type `chat_with_bot()` to start interactive chat!

## 💬 Example Questions

- "What is the total annual budget?"
- "What are the debt limits?"
- "What is the travel policy for domestic trips?"
- "How often are budget reviews conducted?"
- "What are the emergency fund requirements?"

## 🔧 Features

✅ **Document Processing**: Loads and chunks financial policy  
✅ **Vector Search**: Finds relevant information using embeddings  
✅ **Conversation Memory**: Remembers what you asked before  
✅ **Source Tracking**: Shows which document sections were used  
✅ **Completely Free**: No external APIs or paid services  
✅ **Works Offline**: Everything runs locally in Colab  

## 🎯 Assessment Requirements Met

1. **✅ Extract Data**: Financial policy document loaded and processed
2. **✅ Database Setup**: ChromaDB vector database for search
3. **✅ Build Chatbot**: 
   - Understands questions about the document
   - Uses conversation memory
   - Gives answers from knowledge base
   - Clear and helpful responses

## 🚨 Troubleshooting

**If packages don't install:**
```python
!pip install --upgrade pip
!pip install langchain chromadb sentence-transformers transformers torch
```

**If you get errors:**
- Make sure to run cells in order
- Restart runtime if needed
- Check that all packages installed correctly

## 💡 How It Works

1. **Document Loading**: Financial policy is loaded and split into chunks
2. **Vector Embeddings**: Each chunk is converted to numerical vectors
3. **Search**: Your question is converted to a vector and matched with similar chunks
4. **Answer Generation**: Simple logic generates answers based on keywords
5. **Memory**: Conversation history is stored for context

## 🎉 That's It!

- **No API keys needed**
- **No external services**
- **Completely free**
- **Works in Google Colab**
- **Meets all assessment requirements**

Just copy the code, run it in Colab, and start asking questions about the financial policy!