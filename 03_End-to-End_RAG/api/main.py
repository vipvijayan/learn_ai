#!/usr/bin/env python3
"""
FastAPI Backend for PDF and Excel RAG System
Handles file uploads and provides RAG query endpoints
"""

# Required packages (install with: pip install fastapi uvicorn python-multipart python-dotenv)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import sys
import numpy as np
import PyPDF2
from datetime import datetime
import pandas as pd
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

# Import aimakerspace modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the parent directory containing aimakerspace to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.vectordatabase import VectorDatabase

# Load environment variables
load_dotenv()

# RAG Templates from the notebook
RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

app = FastAPI(title="PDF & Excel RAG API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        "https://vercel.app"
    ],  # React frontend and Vercel domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI models
try:
    # Check if API key is loaded from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    print(f"🔑 Found API Key: {openai_api_key[:15]}...")
    
    # Test with direct OpenAI client first
    print("🧪 Testing API key with direct OpenAI client...")
    import openai
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Test a simple API call
    test_response = client.embeddings.create(
        input="test",
        model="text-embedding-3-small"
    )
    print(f"✅ Direct OpenAI API test successful: {len(test_response.data[0].embedding)} dimensions")
    
    # Ensure environment variable is set for aimakerspace
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    print("🚀 Initializing aimakerspace ChatOpenAI...")
    chat_openai = ChatOpenAI()
    print("✅ ChatOpenAI initialized")
    
    print("🚀 Initializing aimakerspace EmbeddingModel...")
    embedding_model = EmbeddingModel()
    print("✅ EmbeddingModel initialized")
    
    # Test aimakerspace embedding
    print("🧪 Testing aimakerspace EmbeddingModel...")
    test_embedding = embedding_model.get_embedding("hello world")
    print(f"✅ Aimakerspace embedding test successful: {len(test_embedding) if test_embedding else 0} dimensions")
    
    print("🎉 All OpenAI models initialized and tested successfully!")
    
except Exception as e:
    print(f"❌ Error initializing OpenAI models: {e}")
    import traceback
    print(f"❌ Full traceback: {traceback.format_exc()}")
    chat_openai = None
    embedding_model = None
    print("📝 Falling back to basic keyword search functionality")

# Global storage for vector databases and metadata
vector_databases = {}
metadata_mappings = {}
uploaded_files = {}
rag_pipelines = {}

# RAG Prompt templates
rag_system_prompt = SystemRolePrompt(
    RAG_SYSTEM_TEMPLATE,
    strict=True,
    defaults={
        "response_style": "concise",
        "response_length": "brief"
    }
)

rag_user_prompt = UserRolePrompt(
    RAG_USER_TEMPLATE,
    strict=True,
    defaults={
        "context_count": "",
        "similarity_scores": ""
    }
)

# RAG Classes from the notebook
class EnhancedPDFLoader:
    def __init__(self):
        pass
    
    def load_pdf_with_metadata(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Load PDF with enhanced metadata extraction"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        documents = []
        metadata_list = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        
                        if text.strip():
                            documents.append(text)
                            
                            page_metadata = {
                                "source": "PDF Document",
                                "file_name": os.path.basename(file_path),
                                "page_number": page_num + 1,
                                "total_pages": total_pages,
                                "page_content_length": len(text),
                                "content_type": "PDF Page",
                                "extraction_date": datetime.now().isoformat()
                            }
                            metadata_list.append(page_metadata)
                            
                    except Exception as page_error:
                        continue
                
                return documents, metadata_list
                
        except Exception as e:
            raise Exception(f"Error loading PDF: {e}")

class FastPDFTextSplitter:
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_pdf_texts_with_metadata(self, texts: List[str], metadata_list: List[Dict] = None) -> Tuple[List[str], List[Dict]]:
        """Split PDF texts while preserving metadata"""
        split_texts = []
        split_metadata = []
        
        for i, text in enumerate(texts):
            base_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            start = 0
            chunk_num = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                
                if chunk.strip():
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_number": chunk_num,
                        "chunk_length": len(chunk),
                        "content_type": "PDF Page Chunk"
                    })
                    
                    split_texts.append(chunk)
                    split_metadata.append(chunk_metadata)
                    chunk_num += 1
                
                start += (self.chunk_size - self.overlap)
                if start >= len(text):
                    break
        
        return split_texts, split_metadata

class ExcelContentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        
    def extract_text_from_dataframe(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convert DataFrame to structured text format"""
        text_content = f"Sheet: {sheet_name}\n"
        text_content += f"Columns: {', '.join(df.columns.astype(str))}\n"
        text_content += f"Number of rows: {len(df)}\n\n"
        
        text_content += "Column Data Types:\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            text_content += f"- {col}: {dtype} ({non_null_count} non-null values)\n"
        text_content += "\n"
        
        text_content += "Sample Data (First 10 Rows):\n"
        sample_df = df.head(10)
        
        for idx, row in sample_df.iterrows():
            text_content += f"Row {idx + 1}:\n"
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    text_content += f"  {col}: {value}\n"
            text_content += "\n"
        
        return text_content
    
    def load_excel_content(self) -> Tuple[List[str], List[Dict]]:
        """Load Excel content with comprehensive metadata"""
        try:
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names
            
            documents = []
            metadata_list = []
            
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                    
                    if df.empty:
                        continue
                    
                    text_content = self.extract_text_from_dataframe(df, sheet_name)
                    documents.append(text_content)
                    
                    sheet_metadata = {
                        "source": "Excel File",
                        "filename": self.filename,
                        "sheet_name": sheet_name,
                        "total_sheets": len(sheet_names),
                        "sheet_index": sheet_names.index(sheet_name),
                        "row_count": df.shape[0],
                        "column_count": df.shape[1],
                        "columns": list(df.columns.astype(str)),
                        "content_type": "Excel Spreadsheet",
                        "extraction_method": "pandas"
                    }
                    
                    metadata_list.append(sheet_metadata)
                    
                except Exception as sheet_error:
                    continue
            
            return documents, metadata_list
            
        except Exception as e:
            raise Exception(f"Error loading Excel content: {e}")

class ExcelTextSplitter:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_excel_content(self, texts: List[str], metadata_list: List[Dict] = None) -> Tuple[List[str], List[Dict]]:
        """Split Excel content while preserving sheet and row context"""
        split_texts = []
        split_metadata = []
        
        for i, text in enumerate(texts):
            base_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            sheet_sections = text.split('\n\n')
            current_chunk = ""
            chunk_num = 0
            
            for section in sheet_sections:
                if not section.strip():
                    continue
                    
                if len(current_chunk + section) > self.chunk_size and current_chunk:
                    split_texts.append(current_chunk.strip())
                    
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_number": chunk_num,
                        "chunk_length": len(current_chunk),
                        "original_doc_index": i,
                        "content_type": "Excel Spreadsheet Chunk"
                    })
                    split_metadata.append(chunk_metadata)
                    
                    chunk_num += 1
                    current_chunk = section
                else:
                    current_chunk += "\n" + section if current_chunk else section
            
            if current_chunk.strip():
                split_texts.append(current_chunk.strip())
                
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_number": chunk_num,
                    "chunk_length": len(current_chunk),
                    "original_doc_index": i,
                    "content_type": "Excel Spreadsheet Chunk"
                })
                split_metadata.append(chunk_metadata)
        
        return split_texts, split_metadata

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> dict:
        # Retrieve relevant contexts
        context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
        
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)

        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    file_type: Optional[str] = None
    file_name: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    file_type: str
    query: str

class FileInfo(BaseModel):
    name: str
    type: str
    size: int
    status: str

def process_pdf_file(file_path: str, file_name: str):
    """Process PDF file and create vector database using enhanced RAG implementation"""
    try:
        # Use enhanced PDF loader
        pdf_loader = EnhancedPDFLoader()
        documents, metadata = pdf_loader.load_pdf_with_metadata(file_path)
        
        # Split documents
        splitter = FastPDFTextSplitter(chunk_size=300, overlap=50)
        split_documents, split_metadata = splitter.split_pdf_texts_with_metadata(documents, metadata)
        
        # Limit chunks for performance (use first 10)
        max_chunks = min(10, len(split_documents))
        limited_documents = split_documents[:max_chunks]
        limited_metadata = split_metadata[:max_chunks]
        
        # Only create vector database and RAG pipeline if OpenAI models are available
        if embedding_model and chat_openai:
            # Create vector database using aimakerspace VectorDatabase
            vector_db = VectorDatabase(embedding_model=embedding_model)
            
            # Insert documents into vector database
            for i, document in enumerate(limited_documents):
                embedding = embedding_model.get_embedding(document)
                vector_db.insert(document, np.array(embedding))
            
            # Create RAG pipeline
            rag_pipeline = RetrievalAugmentedQAPipeline(
                llm=chat_openai,
                vector_db_retriever=vector_db,
                response_style="detailed",
                include_scores=True
            )
            
            # Store in global storage
            vector_databases[file_name] = vector_db
            rag_pipelines[file_name] = rag_pipeline
        
        # Store metadata mapping regardless of OpenAI availability
        metadata_mappings[file_name] = {doc: meta for doc, meta in zip(limited_documents, limited_metadata)}
        
        result = {
            "status": "success", 
            "chunks": len(limited_documents), 
            "pages": len(documents),
            "total_chunks_available": len(split_documents)
        }
        
        if not embedding_model or not chat_openai:
            result["warning"] = "OpenAI models not available. File processed but RAG functionality limited. Please set OPENAI_API_KEY environment variable."
            
        return result
        
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def process_excel_file(file_path: str, file_name: str):
    """Process Excel file and create vector database using enhanced RAG implementation"""
    try:
        # Use enhanced Excel loader
        excel_loader = ExcelContentLoader(file_path)
        documents, metadata = excel_loader.load_excel_content()
        
        # Split documents
        splitter = ExcelTextSplitter(chunk_size=400, overlap=50)
        split_documents, split_metadata = splitter.split_excel_content(documents, metadata)
        
        # Limit chunks for performance (use first 8)
        max_chunks = min(8, len(split_documents))
        limited_documents = split_documents[:max_chunks]
        limited_metadata = split_metadata[:max_chunks]
        
        # Only create vector database and RAG pipeline if OpenAI models are available
        if embedding_model and chat_openai:
            # Create vector database using aimakerspace VectorDatabase
            vector_db = VectorDatabase(embedding_model=embedding_model)
            
            # Insert documents into vector database
            for i, document in enumerate(limited_documents):
                embedding = embedding_model.get_embedding(document)
                vector_db.insert(document, np.array(embedding))
            
            # Create RAG pipeline
            rag_pipeline = RetrievalAugmentedQAPipeline(
                llm=chat_openai,
                vector_db_retriever=vector_db,
                response_style="detailed",
                include_scores=True
            )
            
            # Store in global storage
            vector_databases[file_name] = vector_db
            rag_pipelines[file_name] = rag_pipeline
        
        # Store metadata mapping regardless of OpenAI availability
        metadata_mappings[file_name] = {doc: meta for doc, meta in zip(limited_documents, limited_metadata)}
        
        result = {
            "status": "success", 
            "chunks": len(limited_documents), 
            "sheets": len(documents),
            "total_chunks_available": len(split_documents)
        }
        
        if not embedding_model or not chat_openai:
            result["warning"] = "OpenAI models not available. File processed but RAG functionality limited. Please set OPENAI_API_KEY environment variable."
            
        return result
        
    except Exception as e:
        raise Exception(f"Error processing Excel: {str(e)}")

@app.get("/")
async def root():
    return {"message": "PDF & Excel RAG API is running!"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process PDF or Excel files"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file type
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ['.pdf', '.xlsx', '.xls']:
        raise HTTPException(status_code=400, detail="Only PDF and Excel files are supported")
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Process based on file type
        if file_extension == '.pdf':
            result = process_pdf_file(tmp_path, file.filename)
            file_type = "PDF"
        else:  # Excel files
            result = process_excel_file(tmp_path, file.filename)
            file_type = "Excel"
        
        # Store file info
        uploaded_files[file.filename] = {
            "name": file.filename,
            "type": file_type,
            "size": file.size or 0,
            "status": "processed",
            "result": result
        }
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return JSONResponse(content={
            "message": f"{file_type} file uploaded and processed successfully",
            "filename": file.filename,
            "type": file_type,
            "result": result
        })
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

def handle_fallback_search(request: QueryRequest):
    """Handle search when OpenAI models aren't available - provides basic text search"""
    
    query_lower = request.query.lower()
    results = []
    
    # Search through uploaded file metadata for basic keyword matching
    for file_name, file_info in uploaded_files.items():
        if request.file_name and request.file_name != file_name:
            continue
            
        file_type = file_info.get("type", "Unknown")
        
        # Search through metadata mappings for this file
        if file_name in metadata_mappings:
            for text, metadata in metadata_mappings[file_name].items():
                # Simple keyword search in document text
                if any(keyword.strip() in text.lower() for keyword in query_lower.split() if keyword.strip()):
                    results.append({
                        "text": text[:300] + "..." if len(text) > 300 else text,
                        "score": 0.0,  # No semantic similarity without embeddings
                        "metadata": metadata,
                        "file_name": file_name,
                        "file_type": file_type
                    })
    
    if not results:
        # If no keyword matches found, return some sample content from files
        for file_name, file_info in uploaded_files.items():
            if request.file_name and request.file_name != file_name:
                continue
                
            file_type = file_info.get("type", "Unknown")
            
            if file_name in metadata_mappings:
                # Return first few chunks as sample
                sample_texts = list(metadata_mappings[file_name].items())[:3]
                for text, metadata in sample_texts:
                    results.append({
                        "text": text[:300] + "..." if len(text) > 300 else text,
                        "score": 0.0,
                        "metadata": metadata,
                        "file_name": file_name,
                        "file_type": file_type
                    })
    
    # Limit results
    results = results[:5]
    
    if not results:
        raise HTTPException(status_code=404, detail="No content found in uploaded documents")
    
    # Create fallback response
    response_text = f"⚠️ **Limited Search Results** (OpenAI API not configured)\n\n"
    response_text += f"Found {len(results)} text sections containing your search terms.\n\n"
    response_text += "**Note:** This is a basic keyword search. For AI-powered semantic search and detailed answers, please:\n"
    response_text += "1. Set the OPENAI_API_KEY environment variable\n"
    response_text += "2. Restart the server\n\n"
    response_text += "**Search Results:**\n"
    
    for i, result in enumerate(results, 1):
        response_text += f"{i}. {result['text'][:150]}...\n\n"
    
    return QueryResponse(
        response=response_text,
        sources=results,
        file_type=results[0]["file_type"] if results else "Unknown",
        query=request.query
    )

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """Query uploaded documents using enhanced RAG pipeline"""
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Check if we have any uploaded files
    if not uploaded_files:
        raise HTTPException(status_code=404, detail="No documents uploaded yet")
    
    # If OpenAI models aren't available, provide fallback search functionality
    if not chat_openai or not embedding_model:
        return handle_fallback_search(request)
    
    # If specific file requested
    if request.file_name and request.file_name in rag_pipelines:
        rag_pipeline = rag_pipelines[request.file_name]
        file_info = uploaded_files.get(request.file_name, {})
        file_type = file_info.get("type", "Unknown")
        
        try:
            # Use the RAG pipeline to get a proper response
            result = rag_pipeline.run_pipeline(
                user_query=request.query,
                k=3,
                response_length="detailed"
            )
            
            # Format sources for API response
            sources = []
            for i, (text, score) in enumerate(result["context"], 1):
                metadata = metadata_mappings.get(request.file_name, {}).get(text, {})
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "score": float(score),
                    "metadata": metadata
                })
            
            return QueryResponse(
                response=result["response"],
                sources=sources,
                file_type=file_type,
                query=request.query
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")
    
    # Query all documents using combined approach
    elif rag_pipelines:
        # Combine all vector databases for multi-source search
        all_contexts = []
        all_sources = []
        
        for file_name, rag_pipeline in rag_pipelines.items():
            try:
                # Get contexts from this file's RAG pipeline
                vector_db = vector_databases.get(file_name)
                if vector_db:
                    contexts = vector_db.search_by_text(request.query, k=2)
                    file_info = uploaded_files.get(file_name, {})
                    file_type = file_info.get("type", "Unknown")
                    
                    for text, score in contexts:
                        all_contexts.append((text, score))
                        metadata = metadata_mappings.get(file_name, {}).get(text, {})
                        all_sources.append({
                            "text": text[:200] + "..." if len(text) > 200 else text,
                            "score": float(score),
                            "metadata": metadata,
                            "file_name": file_name,
                            "file_type": file_type
                        })
            except Exception as e:
                continue
        
        if not all_contexts:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Sort by similarity score and take top results
        all_contexts.sort(key=lambda x: x[1], reverse=False)  # Lower distance is better
        top_contexts = all_contexts[:5]
        
        # Create a combined context for RAG response
        context_prompt = ""
        for i, (context, score) in enumerate(top_contexts, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
        
        # Use RAG prompts to generate response
        system_params = {
            "response_style": "detailed",
            "response_length": "comprehensive"
        }
        
        user_params = {
            "user_query": request.query,
            "context": context_prompt.strip(),
            "context_count": len(top_contexts),
            "similarity_scores": f"Relevance scores: {', '.join([f'Source {i+1}: {score:.3f}' for i, (_, score) in enumerate(top_contexts)])}"
        }
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)
        
        # Get response from ChatOpenAI
        response = chat_openai.run([formatted_system_prompt, formatted_user_prompt])
        
        return QueryResponse(
            response=response,
            sources=all_sources[:5],
            file_type="Multiple",
            query=request.query
        )
    
    else:
        raise HTTPException(status_code=404, detail="No documents uploaded yet")

@app.get("/api/files")
async def list_files():
    """List all uploaded files"""
    return {"files": list(uploaded_files.values())}

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific file"""
    if filename not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Remove from storage
    if filename in vector_databases:
        del vector_databases[filename]
    if filename in metadata_mappings:
        del metadata_mappings[filename]
    if filename in uploaded_files:
        del uploaded_files[filename]
    
    return {"message": f"File '{filename}' deleted successfully"}

# Health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "uploaded_files": len(uploaded_files),
        "vector_databases": len(vector_databases)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)