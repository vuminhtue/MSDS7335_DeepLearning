#!/usr/bin/env python3
"""
TueChatRag.py - RAG Chatbot with Ollama, Langchain, ChromaDB, and Gradio

INSTALLATION:
1. Install Ollama: https://ollama.ai/
2. Pull required models:
   ollama pull gemma3:1b
   ollama pull mxbai-embed-large:335m
   ollama pull deepseek-r1:1.5b
   ollama pull llama3.2:1b
   ollama pull qwen2.5vl:3b
3. Install Python dependencies:
   pip install -r requirements.txt
4. Run the application:
   python TueChatRag.py

USAGE:
- Upload PDF or text files using the upload button
- Select your preferred LLM model from the dropdown
- Adjust temperature for response creativity (0=focused, 1=creative)
- Ask questions about your uploaded documents in the chat interface
- The system will use RAG to provide contextual answers based on your documents

FEATURES:
- Local LLM inference with Ollama (no API calls)
- Document embedding and vector search with ChromaDB
- Multiple file format support (PDF, TXT)
- Real-time model switching
- Temperature control for response tuning
- Clean ChromaDB storage management
"""

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional
import logging

import gradio as gr
import chromadb
from chromadb.config import Settings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatBot:
    def __init__(self):
        """Initialize the RAG ChatBot"""
        self.chroma_db_path = "./chromadb"
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:335m",
            base_url="http://localhost:11434"
        )
        
        # Available LLM models
        self.available_models = [
            "gemma3:1b",
            "deepseek-r1:1.5b", 
            "llama3.2:1b",
            "qwen2.5vl:3b"
        ]
        
        # Initialize with default model
        self.current_llm = Ollama(
            model="gemma3:1b",
            base_url="http://localhost:11434",
            temperature=0.5
        )
        
        # Setup ChromaDB client
        self.setup_chromadb()
        
    def setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        try:
            # Clear any existing client reference
            self.chroma_client = None
            
            # Ensure directory exists
            os.makedirs(self.chroma_db_path, exist_ok=True)
            
            # Create ChromaDB client with proper settings
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            # Fallback to in-memory client if persistent fails
            try:
                self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
                logger.info("Fallback to in-memory ChromaDB client")
            except Exception as fallback_e:
                logger.error(f"Error with fallback ChromaDB setup: {fallback_e}")
            
    def clean_chromadb(self):
        """Clean up existing ChromaDB data"""
        try:
            # Reset vectorstore and QA chain
            self.vectorstore = None
            self.qa_chain = None
            
            # Close existing client connection
            if hasattr(self, 'chroma_client') and self.chroma_client:
                try:
                    # Try to reset the client if it supports it
                    if hasattr(self.chroma_client, 'reset'):
                        self.chroma_client.reset()
                except:
                    pass
                self.chroma_client = None
            
            # Remove the persistent directory
            if os.path.exists(self.chroma_db_path):
                shutil.rmtree(self.chroma_db_path)
                logger.info("Cleaned up existing ChromaDB data")
            
            # Wait a moment for filesystem operations to complete
            time.sleep(0.5)
            
            # Recreate the ChromaDB setup
            self.setup_chromadb()
            
        except Exception as e:
            logger.error(f"Error cleaning ChromaDB: {e}")
            # Force reinitialize if cleanup fails
            try:
                self.chroma_client = None
                self.setup_chromadb()
            except Exception as reinit_e:
                logger.error(f"Error reinitializing ChromaDB: {reinit_e}")
            
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
            
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {txt_path}: {e}")
            return ""
            
    def process_uploaded_files(self, files: List) -> str:
        """Process uploaded files and create vector embeddings"""
        if not files:
            return "No files uploaded."
        
        try:
            # Clean up previous data
            self.clean_chromadb()
            
            documents = []
            processed_files = []
            
            for file in files:
                if file is None:
                    continue
                    
                file_path = file.name
                file_name = os.path.basename(file_path)
                
                # Extract text based on file type
                if file_path.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.lower().endswith(('.txt', '.text')):
                    text = self.extract_text_from_txt(file_path)
                else:
                    continue
                    
                if text.strip():
                    # Create document object
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_name}
                    )
                    documents.append(doc)
                    processed_files.append(file_name)
                    
            if not documents:
                return "No valid content extracted from uploaded files."
                
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} document chunks")
            
            # Create vector store with unique collection name to avoid conflicts
            collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
            
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_path,
                collection_name=collection_name,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create QA chain
            self.create_qa_chain()
            
            return f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}\nCreated {len(splits)} text chunks for RAG."
            
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return f"Error processing files: {str(e)}"
            
    def create_qa_chain(self):
        """Create the QA chain for RAG"""
        if not self.vectorstore:
            return
            
        try:
            # Create custom prompt template
            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.current_llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            logger.info("QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            
    def update_llm_model(self, model_name: str, temperature: float):
        """Update the LLM model and temperature"""
        try:
            self.current_llm = Ollama(
                model=model_name,
                base_url="http://localhost:11434",
                temperature=temperature
            )
            
            # Recreate QA chain with new LLM
            if self.vectorstore:
                self.create_qa_chain()
                
            logger.info(f"Updated LLM to {model_name} with temperature {temperature}")
            return f"Updated model to {model_name} (temperature: {temperature})"
            
        except Exception as e:
            logger.error(f"Error updating LLM: {e}")
            return f"Error updating model: {str(e)}"
            
    def chat(self, question: str, history: List) -> tuple:
        """Process chat question and return response"""
        if not question.strip():
            return history, ""
            
        if not self.qa_chain:
            error_msg = "Please upload and process files first before asking questions."
            history.append([question, error_msg])
            return history, ""
            
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            # Add source information if available
            if "source_documents" in result and result["source_documents"]:
                sources = list(set([doc.metadata.get("source", "Unknown") 
                                 for doc in result["source_documents"]]))
                answer += f"\n\n**Sources:** {', '.join(sources)}"
            
            history.append([question, answer])
            return history, ""
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_msg = f"Error processing question: {str(e)}"
            history.append([question, error_msg])
            return history, ""

def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Initialize chatbot
    chatbot = RAGChatBot()
    
    def upload_files(files):
        """Handle file upload"""
        return chatbot.process_uploaded_files(files)
        
    def update_model_settings(model_name, temperature):
        """Handle model and temperature updates"""
        return chatbot.update_llm_model(model_name, temperature)
        
    def chat_response(message, history):
        """Handle chat interaction"""
        return chatbot.chat(message, history)
        
    # Create Gradio interface
    with gr.Blocks(title="TueChatRag - RAG Chatbot", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🤖 TueChatRag - RAG Chatbot")
        gr.Markdown("Upload documents and chat with them using local Ollama models!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                gr.Markdown("### 📁 Upload Documents")
                file_upload = gr.File(
                    label="Upload PDF or Text files",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".text"]
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    max_lines=3
                )
                
                # Model settings section
                gr.Markdown("### ⚙️ Model Settings")
                model_dropdown = gr.Dropdown(
                    choices=chatbot.available_models,
                    value="gemma3:1b",
                    label="Select LLM Model"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Temperature"
                )
                
                model_status = gr.Textbox(
                    label="Model Status",
                    interactive=False,
                    max_lines=2
                )
                
            with gr.Column(scale=2):
                # Chat section
                gr.Markdown("### 💬 Chat Interface")
                chatbot_interface = gr.Chatbot(
                    label="Chat with your documents",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="Type your question here...",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")
                    
                clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        file_upload.change(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        # Update model settings when dropdown or slider changes
        model_dropdown.change(
            fn=update_model_settings,
            inputs=[model_dropdown, temperature_slider],
            outputs=[model_status]
        )
        
        temperature_slider.change(
            fn=update_model_settings,
            inputs=[model_dropdown, temperature_slider],
            outputs=[model_status]
        )
        
        # Chat functionality
        submit_btn.click(
            fn=chat_response,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface, msg_input]
        )
        
        msg_input.submit(
            fn=chat_response,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface, msg_input]
        )
        
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot_interface, msg_input]
        )
        
    return demo

def main():
    """Main function to run the application"""
    try:
        # Check if required directories exist
        os.makedirs("./chromadb", exist_ok=True)
        
        print("🚀 Starting TueChatRag...")
        print("📋 Make sure Ollama is running with the required models:")
        print("   - gemma3:1b")
        print("   - mxbai-embed-large:335m") 
        print("   - deepseek-r1:1.5b")
        print("   - llama3.2:1b")
        print("   - qwen2.5vl:3b")
        print("\n🌐 Creating Gradio interface...")
        
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down TueChatRag...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()