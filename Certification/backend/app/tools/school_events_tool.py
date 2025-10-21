"""
Custom LangChain Tool for searching school events and programs
"""
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class SchoolEventsSearchInput(BaseModel):
    """Input for the School Events Search tool."""
    query: str = Field(description="The search query to find relevant school events and programs")


class SchoolEventsSearchTool(BaseTool):
    """Tool for searching school events and programs from email data."""
    
    name: str = "school_events_search"
    description: str = """
    Useful for finding information about school events, programs, and activities.
    Use this tool when parents ask about:
    - Coding programs or technology classes
    - Art classes or creative programs
    - Sports activities or athletic programs
    - Music programs or performances
    - Holiday camps or day camps
    - Registration dates and deadlines
    - Age requirements or grade levels
    - Program schedules and timing
    - Spring break, winter break, or other school holidays
    - District-specific events (Leander, Round Rock, Harmony, etc.)
    
    Input should be a search query describing what kind of event or program information is needed.
    """
    args_schema: Type[BaseModel] = SchoolEventsSearchInput
    
    vector_store: Optional[Qdrant] = None
    
    def __init__(self):
        super().__init__()
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Initialize the vector store with school events data using Qdrant"""
        try:
            # Load school events data from text files
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            documents = []
            # Load all TXT files from data directory
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            metadata = {
                                "source": filename,
                                "type": "school_event"
                            }
                            documents.append(Document(page_content=content, metadata=metadata))
            if not documents:
                print("Warning: No TXT files found in data directory")
                return
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=embeddings,
                location=":memory:"
            )
            print(f"✅ School Events Tool initialized with {len(documents)} documents and {len(chunks)} chunks (Qdrant)")
        except Exception as e:
            print(f"❌ Error initializing School Events Tool: {e}")
            raise e
    
    def _run(self, query: str) -> str:
        """Search for school events based on the query"""
        logger.info(f"\n🔍 TOOL INVOKED: school_events_search")
        logger.info(f"   Query: {query}")
        
        start_time = datetime.now()
        
        try:
            if not self.vector_store:
                logger.error("   ❌ Vector store not initialized")
                return "Error: School events database is not initialized."
            
            # Perform similarity search
            logger.info(f"   🔎 Performing similarity search (k=3)...")
            results = self.vector_store.similarity_search(query, k=3)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if not results:
                logger.warning(f"   ⚠️  No results found (took {duration:.2f}s)")
                return "No relevant school events or programs found for your query."
            
            logger.info(f"   ✅ Found {len(results)} results (took {duration:.2f}s)")
            for i, doc in enumerate(results, 1):
                logger.info(f"      Result {i}: {doc.metadata.get('source', 'Unknown')} - {len(doc.page_content)} chars")
            
            # Format results
            response_parts = []
            response_parts.append(f"Found {len(results)} relevant school events/programs:\n")
            
            for i, doc in enumerate(results, 1):
                response_parts.append(f"\n📋 Result {i}:")
                response_parts.append(doc.page_content)
                response_parts.append(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
                if i < len(results):
                    response_parts.append("\n" + "-" * 70)
            
            response = "\n".join(response_parts)
            logger.info(f"   📄 Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"   ❌ Error in tool execution: {str(e)}")
            return f"Error searching school events: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the run method"""
        return self._run(query)


def create_school_events_tool():
    """Create and return an instance of the School Events Search Tool"""
    return SchoolEventsSearchTool()
