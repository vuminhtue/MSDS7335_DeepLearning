# TueChatRag - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Ollama, Langchain, ChromaDB, and Gradio for local document querying.

## Features

- 🤖 **Local LLM inference** using Ollama (no API keys required)
- 📚 **Document upload** support for PDF and text files
- 🔍 **Vector search** with ChromaDB for efficient document retrieval
- 🎯 **Multiple LLM models** with real-time switching
- 🌡️ **Temperature control** for response creativity
- 🖥️ **User-friendly GUI** built with Gradio

## Quick Start

### 1. Install Ollama

Download and install Ollama from: https://ollama.ai/

### 2. Pull Required Models

```bash
ollama pull gemma2:1b
ollama pull mxbai-embed-large:335m
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1b
ollama pull qwen2.5vl:3b
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python TueChatRag.py
```

The application will start on `http://localhost:7860`

## Usage

1. **Upload Documents**: Use the file upload button to add PDF or text files
2. **Select Model**: Choose from the available LLM models in the dropdown
3. **Adjust Temperature**: Use the slider to control response creativity (0=focused, 1=creative)
4. **Ask Questions**: Type questions about your documents in the chat interface
5. **Get Answers**: The system will provide contextual answers with source references

## Supported Models

- gemma2:1b (default)
- deepseek-r1:1.5b
- llama3.2:1b
- qwen2.5vl:3b

## Technical Details

- **RAG Framework**: Langchain
- **Vector Database**: ChromaDB
- **Embedding Model**: mxbai-embed-large:335m
- **Text Splitting**: Recursive character splitting (1000 chars, 200 overlap)
- **Retrieval**: Top-4 similar chunks
- **GUI**: Gradio web interface

## File Structure

```
FinalProject/
├── TueChatRag.py          # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── chromadb/             # Vector database storage (auto-created)
```

## Notes

- ChromaDB data is automatically cleaned when new files are uploaded
- Make sure Ollama is running before starting the application
- The application runs locally and doesn't require internet connectivity after initial model downloads