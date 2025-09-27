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
from typing import List, Optional, Dict, Any
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import sys

# Import our RAG implementations
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

app = FastAPI(title="PDF & Excel RAG API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for vector databases and metadata
vector_databases = {}
metadata_mappings = {}
uploaded_files = {}

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
    """Process PDF file and create vector database"""
    try:
        import PyPDF2
        import numpy as np
        
        # Enhanced PDF Loader
        class EnhancedPDFLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
                
            def load_pdf(self):
                try:
                    with open(self.file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        documents = []
                        metadata_list = []
                        
                        for page_num, page in enumerate(pdf_reader.pages):
                            try:
                                text = page.extract_text()
                                if text.strip():
                                    documents.append(text)
                                    metadata_list.append({
                                        "source": os.path.basename(self.file_path),
                                        "page_number": page_num + 1,
                                        "total_pages": len(pdf_reader.pages),
                                        "content_type": "PDF Document",
                                        "file_name": file_name
                                    })
                            except Exception as page_error:
                                continue
                        
                        return documents, metadata_list
                except Exception as e:
                    raise Exception(f"Error loading PDF: {e}")
        
        # PDF Text Splitter
        class PDFTextSplitter:
            def __init__(self, chunk_size: int = 500):
                self.chunk_size = chunk_size
            
            def split_text(self, texts, metadata_list):
                split_texts = []
                split_metadata = []
                
                for i, text in enumerate(texts):
                    base_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                    
                    # Simple chunking by sentences
                    sentences = text.split('.')
                    current_chunk = ""
                    chunk_num = 0
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                            split_texts.append(current_chunk.strip())
                            
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "chunk_number": chunk_num,
                                "chunk_length": len(current_chunk),
                                "content_type": "PDF Document Chunk"
                            })
                            split_metadata.append(chunk_metadata)
                            
                            chunk_num += 1
                            current_chunk = sentence + "."
                        else:
                            current_chunk += " " + sentence + "." if current_chunk else sentence + "."
                    
                    if current_chunk.strip():
                        split_texts.append(current_chunk.strip())
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata.update({
                            "chunk_number": chunk_num,
                            "chunk_length": len(current_chunk),
                            "content_type": "PDF Document Chunk"
                        })
                        split_metadata.append(chunk_metadata)
                
                return split_texts, split_metadata
        
        # Simple Vector Database
        class SimpleVectorDatabase:
            def __init__(self, texts, metadata_list):
                self.texts = texts[:10]  # Limit for demo
                self.metadata_list = metadata_list[:10] if metadata_list else []
                self.vectors = []
                self.create_simple_vectors()
            
            def create_simple_vectors(self):
                for text in self.texts:
                    features = [
                        len(text), len(text.split()), text.count('.'), text.count(','),
                        len(set(text.lower().split())), text.count('space'),
                        text.count('exploration'), text.count('mission')
                    ]
                    
                    max_len = max(features) if max(features) > 0 else 1
                    normalized_vector = [f / max_len for f in features]
                    while len(normalized_vector) < 8:
                        normalized_vector.append(0.0)
                    
                    self.vectors.append(np.array(normalized_vector))
            
            def similarity_search(self, query: str, k: int = 3):
                query_features = [
                    len(query), len(query.split()), query.count('.'), query.count(','),
                    len(set(query.lower().split())), query.count('space'),
                    query.count('exploration'), query.count('mission')
                ]
                
                max_len = max(query_features) if max(query_features) > 0 else 1
                query_vector = np.array([f / max_len for f in query_features])
                while len(query_vector) < 8:
                    query_vector = np.append(query_vector, 0.0)
                
                similarities = []
                for i, doc_vector in enumerate(self.vectors):
                    dot_product = np.dot(query_vector, doc_vector)
                    norm_query = np.linalg.norm(query_vector)
                    norm_doc = np.linalg.norm(doc_vector)
                    
                    if norm_query > 0 and norm_doc > 0:
                        similarity = dot_product / (norm_query * norm_doc)
                    else:
                        similarity = 0.0
                    
                    similarities.append((i, similarity))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_results = similarities[:k]
                
                results = []
                for idx, score in top_results:
                    results.append((self.texts[idx], score, self.metadata_list[idx]))
                
                return results
        
        # Process the PDF
        loader = EnhancedPDFLoader(file_path)
        documents, metadata = loader.load_pdf()
        
        splitter = PDFTextSplitter(chunk_size=400)
        split_documents, split_metadata = splitter.split_text(documents, metadata)
        
        vector_db = SimpleVectorDatabase(split_documents, split_metadata)
        
        # Store in global storage
        vector_databases[file_name] = vector_db
        metadata_mappings[file_name] = {doc: meta for doc, meta in zip(split_documents, split_metadata)}
        
        return {"status": "success", "chunks": len(split_documents), "pages": len(documents)}
        
    except ImportError:
        raise Exception("Required packages not installed. Please install: pip install PyPDF2 numpy")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def process_excel_file(file_path: str, file_name: str):
    """Process Excel file and create vector database"""
    try:
        import pandas as pd
        import numpy as np
        
        # Excel Content Loader
        class ExcelContentLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
                
            def load_excel_content(self):
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
                            
                            # Extract text content
                            text_content = f"Sheet: {sheet_name}\n"
                            text_content += f"Columns: {', '.join(df.columns.astype(str))}\n"
                            text_content += f"Number of rows: {len(df)}\n\n"
                            
                            # Add sample data
                            sample_df = df.head(10)
                            for idx, row in sample_df.iterrows():
                                text_content += f"Row {idx + 1}:\n"
                                for col in df.columns:
                                    value = row[col]
                                    if pd.notna(value):
                                        text_content += f"  {col}: {value}\n"
                                text_content += "\n"
                            
                            documents.append(text_content)
                            
                            metadata_list.append({
                                "source": "Excel File",
                                "sheet_name": sheet_name,
                                "row_count": df.shape[0],
                                "column_count": df.shape[1],
                                "content_type": "Excel Spreadsheet",
                                "file_name": file_name
                            })
                            
                        except Exception as sheet_error:
                            continue
                    
                    return documents, metadata_list
                    
                except Exception as e:
                    raise Exception(f"Error loading Excel content: {e}")
        
        # Excel Text Splitter
        class ExcelTextSplitter:
            def __init__(self, chunk_size: int = 400):
                self.chunk_size = chunk_size
            
            def split_excel_content(self, texts, metadata_list):
                split_texts = []
                split_metadata = []
                
                for i, text in enumerate(texts):
                    base_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                    
                    # Split by sections
                    sections = text.split('\n\n')
                    current_chunk = ""
                    chunk_num = 0
                    
                    for section in sections:
                        if not section.strip():
                            continue
                            
                        if len(current_chunk + section) > self.chunk_size and current_chunk:
                            split_texts.append(current_chunk.strip())
                            
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "chunk_number": chunk_num,
                                "chunk_length": len(current_chunk),
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
                            "content_type": "Excel Spreadsheet Chunk"
                        })
                        split_metadata.append(chunk_metadata)
                
                return split_texts, split_metadata
        
        # Simple Excel Vector Database
        class SimpleExcelVectorDatabase:
            def __init__(self, texts, metadata_list):
                self.texts = texts[:8]  # Limit for demo
                self.metadata_list = metadata_list[:8] if metadata_list else []
                self.vectors = []
                self.create_simple_vectors()
            
            def create_simple_vectors(self):
                for text in self.texts:
                    features = [
                        len(text), len(text.split()), text.count('Sheet:'), text.count('Row'),
                        text.count(':'), text.count('columns'), text.count('data'),
                        len(set(text.lower().split()))
                    ]
                    
                    max_len = max(features) if max(features) > 0 else 1
                    normalized_vector = [f / max_len for f in features]
                    while len(normalized_vector) < 8:
                        normalized_vector.append(0.0)
                    
                    self.vectors.append(np.array(normalized_vector))
            
            def similarity_search(self, query: str, k: int = 3):
                query_features = [
                    len(query), len(query.split()), query.count('sheet'), query.count('row'),
                    query.count('data'), query.count('column'), query.count('excel'),
                    len(set(query.lower().split()))
                ]
                
                max_len = max(query_features) if max(query_features) > 0 else 1
                query_vector = np.array([f / max_len for f in query_features])
                while len(query_vector) < 8:
                    query_vector = np.append(query_vector, 0.0)
                
                similarities = []
                for i, doc_vector in enumerate(self.vectors):
                    dot_product = np.dot(query_vector, doc_vector)
                    norm_query = np.linalg.norm(query_vector)
                    norm_doc = np.linalg.norm(doc_vector)
                    
                    if norm_query > 0 and norm_doc > 0:
                        similarity = dot_product / (norm_query * norm_doc)
                    else:
                        similarity = 0.0
                    
                    similarities.append((i, similarity))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_results = similarities[:k]
                
                results = []
                for idx, score in top_results:
                    results.append((self.texts[idx], score, self.metadata_list[idx]))
                
                return results
        
        # Process the Excel file
        loader = ExcelContentLoader(file_path)
        documents, metadata = loader.load_excel_content()
        
        splitter = ExcelTextSplitter(chunk_size=300)
        split_documents, split_metadata = splitter.split_excel_content(documents, metadata)
        
        vector_db = SimpleExcelVectorDatabase(split_documents, split_metadata)
        
        # Store in global storage
        vector_databases[file_name] = vector_db
        metadata_mappings[file_name] = {doc: meta for doc, meta in zip(split_documents, split_metadata)}
        
        return {"status": "success", "chunks": len(split_documents), "sheets": len(documents)}
        
    except ImportError:
        raise Exception("Required packages not installed. Please install: pip install pandas openpyxl numpy")
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

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """Query uploaded documents"""
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # If specific file requested
    if request.file_name and request.file_name in vector_databases:
        vector_db = vector_databases[request.file_name]
        file_info = uploaded_files.get(request.file_name, {})
        file_type = file_info.get("type", "Unknown")
        
        try:
            results = vector_db.similarity_search(request.query, k=3)
            
            # Format response
            sources = []
            response_parts = []
            
            for i, (text, score, metadata) in enumerate(results, 1):
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "score": float(score),
                    "metadata": metadata
                })
                response_parts.append(f"Source {i} (similarity: {score:.3f}): {text[:150]}...")
            
            response_text = f"Based on the {file_type} document '{request.file_name}', here are the most relevant sections:\n\n" + "\n\n".join(response_parts)
            
            return QueryResponse(
                response=response_text,
                sources=sources,
                file_type=file_type,
                query=request.query
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")
    
    # Query all documents
    elif vector_databases:
        all_results = []
        all_sources = []
        
        for file_name, vector_db in vector_databases.items():
            try:
                results = vector_db.similarity_search(request.query, k=2)
                file_info = uploaded_files.get(file_name, {})
                file_type = file_info.get("type", "Unknown")
                
                for text, score, metadata in results:
                    all_results.append((text, score, metadata, file_name, file_type))
                    all_sources.append({
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "score": float(score),
                        "metadata": metadata,
                        "file_name": file_name,
                        "file_type": file_type
                    })
            except Exception as e:
                continue
        
        if not all_results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x[1], reverse=True)
        top_results = all_results[:5]
        
        response_parts = []
        for i, (text, score, metadata, file_name, file_type) in enumerate(top_results, 1):
            response_parts.append(f"Source {i} from {file_type} '{file_name}' (similarity: {score:.3f}): {text[:150]}...")
        
        response_text = f"Here are the most relevant sections from your uploaded documents:\n\n" + "\n\n".join(response_parts)
        
        return QueryResponse(
            response=response_text,
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