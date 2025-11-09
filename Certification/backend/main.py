from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional
import json
import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Directory paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
GENERATED_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_results")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'backend.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("="*80)
logger.info("🚀 Starting School Assistant API")
logger.info("="*80)

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Import custom tool
from app.tools.school_events_tool import create_school_events_tool

# Import multi-agent system
from app.agents.multi_agent_system import create_school_events_agents, query_with_agent, query_with_agent_stream

# Import database functions
from app.database import (
    init_database,
    get_or_create_user,
    get_all_schools,
    update_user_school,
    get_user_with_school,
    set_user_schools,
    get_user_schools,
    get_user_with_schools,
    reset_database,
    # Preferences functions
    set_user_preference,
    get_user_preference,
    get_all_user_preferences,
    # Bookmarks functions
    add_bookmark,
    remove_bookmark,
    get_user_bookmarks,
    # Gmail OAuth functions
    save_user_gmail_token,
    get_user_gmail_token,
    disconnect_user_gmail
)

# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================

# CORS configuration
FRONTEND_URL = "http://localhost:3000"
VERCEL_FRONTEND_URL = "https://frontend-j4vkwnj7q-vipin-vijayan-nairs-projects.vercel.app"

# Retrieval method configuration
RETRIEVAL_METHODS = {
    "original": "Original RAG (k=4, query expansion)",
    "naive": "Naive Retrieval (k=10, LCEL chain from Advanced Retrieval)"
}

# ============================================================
# GLOBAL VARIABLES
# ============================================================

# RAG components
retriever = None
generator_chain = None
agent_tools = None
school_events_tool = None
multi_agents = None

# Active retrieval method
active_retrieval_method = "naive"  # Default to naive retrieval
original_retriever = None
naive_retriever = None
original_chain = None
naive_chain = None

# Store queries and responses for evaluation
evaluation_buffer = []

# ============================================================
# FASTAPI APP SETUP
# ============================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("="*80)
    logger.info("�️  Initializing SQLite Database...")
    init_database()
    logger.info("="*80)
    logger.info("📋 AVAILABLE ENDPOINTS:")
    logger.info("   /api/auth/login → User login with email")
    logger.info("   /api/auth/schools → Get list of schools")
    logger.info("   /api/auth/select-school → Select user's school")
    logger.info("   /query → RAG Query (switchable between Original and Naive)")
    logger.info("   /agent-query → Direct Tool Use (school_events_search)")
    logger.info("   /multi-agent-query → Multi-Agent System (WebSearch + LocalEvents)")
    logger.info("   /evaluate-ragas → RAGAS Evaluation (Faithfulness, Relevancy, Precision, Recall)")
    logger.info("   /retrieval-method → Get/Set active retrieval method")
    logger.info("   /retrieval-methods → List all available retrieval methods")
    logger.info("   /tools → List all available tools")
    logger.info("   /agents → List all available agents")
    logger.info("   /health → Health check")
    logger.info("🔍 Active Retrieval Method: Naive Retrieval (default)")
    logger.info("="*80)
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("👋 Shutting down School Assistant API")

app = FastAPI(title="School Assistant API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, VERCEL_FRONTEND_URL],  # React dev server and Vercel production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PYDANTIC MODELS
# ============================================================

class LoginRequest(BaseModel):
    email: str

class SchoolSelectionRequest(BaseModel):
    email: str
    school_ids: List[int]  # List of selected school IDs
    
    @field_validator('school_ids', mode='before')
    @classmethod
    def validate_school_ids(cls, v):
        logger.info(f"🔍 Validating school_ids: {v} (type: {type(v)})")
        if v is None:
            raise ValueError("school_ids cannot be None")
        if not isinstance(v, list):
            raise ValueError(f"school_ids must be a list, got {type(v)}")
        if len(v) == 0:
            raise ValueError("school_ids cannot be empty")
        # Ensure all elements are integers or can be converted to integers
        try:
            return [int(x) for x in v]
        except (ValueError, TypeError) as e:
            raise ValueError(f"All school_ids must be integers: {e}")

class QueryRequest(BaseModel):
    question: str
    email_suffix: str = None  # Optional single email suffix for Gmail filtering (legacy)
    email_suffixes: List[str] = None  # Optional list of email suffixes for multi-school filtering
    school_district: str = None  # Optional school district name for better search context
    school_districts: List[str] = None  # Optional list of school district names

class QueryResponse(BaseModel):
    answer: str
    context: list[str]

class BookmarkRequest(BaseModel):
    email: str
    bookmark_id: str
    message_type: str
    message_content: str
    message_context: str = None
    message_source: str = None
    message_index: int = None

class RemoveBookmarkRequest(BaseModel):
    email: str
    bookmark_id: str

class PreferenceRequest(BaseModel):
    email: str
    preference_key: str
    preference_value: str

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def setup_agent_tools():
    """Initialize agent tools including custom school events tool"""
    global agent_tools, school_events_tool
    
    try:
        logger.info("🔧 Initializing Agent Tools...")
        print("\n🔧 Initializing Agent Tools...")
        
        # Create custom school events tool
        school_events_tool = create_school_events_tool()
        agent_tools = [school_events_tool]
        logger.info(f"  ✅ Created tool: {school_events_tool.name}")
        
        # Add Tavily search if API key is available
        if os.getenv("TAVILY_API_KEY"):
            tavily_tool = TavilySearchResults(max_results=5)
            agent_tools.append(tavily_tool)
            logger.info(f"  ✅ Created tool: {tavily_tool.name}")
            print("✅ Tavily search tool added")
        else:
            logger.warning("  ⚠️ Tavily API key not found, skipping Tavily tool")
            print("⚠️  Tavily API key not found, skipping Tavily tool")
        
        logger.info(f"✅ Agent tools initialized: {len(agent_tools)} tools")
        logger.info(f"   Available tools: {', '.join([tool.name for tool in agent_tools])}")
        print(f"✅ Agent tools initialized: {len(agent_tools)} tools available")
        print(f"   Tools: {[tool.name for tool in agent_tools]}")
        return agent_tools
        
    except Exception as e:
        logger.error(f"❌ Error initializing agent tools: {e}")
        print(f"❌ Error initializing agent tools: {e}")
        raise e

def setup_rag_pipeline():
    """Initialize RAG pipeline with BOTH Original and Naive Retrieval methods
    
    Creates two complete retrieval pipelines for comparison:
    1. Original: k=4 with query expansion (your original implementation)
    2. Naive: k=10 LCEL chain (Advanced Retrieval notebook)
    """
    global retriever, generator_chain
    global original_retriever, naive_retriever, original_chain, naive_chain
    global active_retrieval_method
    
    logger.info("📝 Setting up RAG pipeline with BOTH retrieval methods...")
    
    # Load the TXT files from data directory
    all_events = []
    sources = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.txt'):
            logger.info(f"  Loading: {filename}")
            with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
                event_text = f.read()
                all_events.append(event_text)
                sources.append(filename)
    
    logger.info(f"✅ Loaded {len(all_events)} events from {len(set(sources))} files")
    
    # Create embeddings using OpenAI's text-embedding-3-small
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    logger.info(f"🔤 Using embedding model: text-embedding-3-small")
    
    # Create documents with text splitting for better retrieval
    from langchain_community.vectorstores import Qdrant
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Create text splitter to chunk large JSON documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ",", " ", ""]
    )
    
    # Create initial documents
    documents = [Document(page_content=text, metadata={"source": source}) 
                 for text, source in zip(all_events, sources)]
    
    # Split documents into smaller chunks for better retrieval
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"📄 Split {len(documents)} documents into {len(split_documents)} chunks")
    # Split documents into smaller chunks for better retrieval
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"📄 Split {len(documents)} documents into {len(split_documents)} chunks")
    
    vectorstore = Qdrant.from_documents(
        documents=split_documents,
        embedding=embeddings,
        location=":memory:"
    )
    logger.info(f"💾 Qdrant vector store created with {len(split_documents)} chunks")
    
    # ============================================================
    # METHOD 1: ORIGINAL RETRIEVAL (k=4, no query expansion in chain)
    # ============================================================
    logger.info("🔧 Creating Original Retrieval method (k=4)...")
    original_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Original prompt template
    original_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant for a school events information system.
        
Based on the following information about school events and programs:

{context}

Please answer this question: {question}

Provide a clear, well-formatted response using the following guidelines:
- Start with a brief overview or direct answer
- Use bullet points (•) for listing multiple items or features
- Use clear section headers when appropriate
- Include specific details like dates, times, locations, age ranges, and costs when available
- Keep paragraphs concise and readable
- If information isn't in the context, politely say so and suggest what information is available

Format your response for easy reading with proper spacing and structure.""")
    
    # Original chain: simple prompt | llm | parser
    original_chain = original_prompt | llm | StrOutputParser()
    logger.info(f"  ✅ Original Retrieval chain created (k=4, simple chain)")
    
    # ============================================================
    # METHOD 2: NAIVE RETRIEVAL (k=10)
    # ============================================================
    logger.info("🔧 Creating Naive Retrieval method (k=10, LCEL)...")
    naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    from langchain_core.runnables import RunnablePassthrough
    from operator import itemgetter
    
    # Naive RAG prompt template (from Advanced Retrieval notebook, Cell 12)
    RAG_TEMPLATE = """\
You are a helpful and kind assistant for a school events information system. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Provide clear, well-formatted responses using these guidelines:
- Start with a direct answer or overview
- Use bullet points (•) for multiple items
- Include specific details (dates, times, locations, ages, costs)
- Keep information organized and easy to scan
- Use proper spacing between sections

Query:
{question}

Context:
{context}
"""
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    # Naive Retrieval Chain using LCEL
    naive_chain = (
        {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm, "context": itemgetter("context")}
    )
    logger.info(f"  ✅ Naive Retrieval chain created (k=10, LCEL pattern)")
    
    # ============================================================
    # SET ACTIVE METHOD
    # ============================================================
    if active_retrieval_method == "original":
        retriever = original_retriever
        generator_chain = original_chain
        logger.info(f"🎯 Active method: Original Retrieval (k=4)")
    else:
        retriever = naive_retriever
        generator_chain = naive_chain
        logger.info(f"🎯 Active method: Naive Retrieval (k=10, LCEL)")
    
    logger.info("✅ RAG pipeline setup complete with BOTH methods")
    logger.info("="*80)

@app.get("/")
async def root():
    return {"message": "School Events RAG API is running!", "status": "healthy"}


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """
    Login with email - creates or retrieves user
    """
    try:
        # Validate email format
        email = request.email.strip().lower()
        if not email or '@' not in email:
            raise HTTPException(status_code=400, detail="Invalid email address")
        
        # Get or create user
        user = get_or_create_user(email)
        
        # Get user with all schools
        user_with_schools = get_user_with_schools(email)
        
        # If user has no schools selected, check if there's a saved preference
        if not user_with_schools.get('schools') or len(user_with_schools['schools']) == 0:
            import json
            saved_schools = get_user_preference(email, "selected_schools")
            if saved_schools:
                try:
                    school_ids = json.loads(saved_schools)
                    if school_ids and len(school_ids) > 0:
                        # Restore the saved school selections
                        set_user_schools(email, school_ids)
                        user_with_schools = get_user_with_schools(email)
                        logger.info(f"🔄 Restored {len(school_ids)} saved school(s) for {email}")
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to restore saved schools: {e}")
        
        return {
            "success": True,
            "user": user_with_schools,
            "message": "Login successful"
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/schools")
async def get_schools_list():
    """
    Get list of all available schools for selection
    """
    try:
        schools = get_all_schools()
        return {
            "success": True,
            "schools": schools
        }
    except Exception as e:
        logger.error(f"Error fetching schools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/select-school")
async def select_school(request: SchoolSelectionRequest):
    """
    Update user's selected schools (supports multiple school selection)
    """
    try:
        logger.info(f"📥 School selection request received:")
        logger.info(f"   Email: {request.email}")
        logger.info(f"   School IDs: {request.school_ids}")
        logger.info(f"   School IDs type: {type(request.school_ids)}")
        
        email = request.email.strip().lower()
        
        # Validate school_ids
        if not request.school_ids or len(request.school_ids) == 0:
            raise HTTPException(status_code=400, detail="Please select at least one school")
        
        # Ensure all IDs are integers
        try:
            school_ids = [int(sid) for sid in request.school_ids]
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid school ID format: {e}")
            raise HTTPException(status_code=400, detail="Invalid school ID format")
        
        # Set multiple schools
        set_user_schools(email, school_ids)
        user_with_schools = get_user_with_schools(email)
        
        # Save selected school IDs as a preference
        import json
        set_user_preference(email, "selected_schools", json.dumps(school_ids))
        logger.info(f"💾 Saved selected schools to preferences: {school_ids}")
        
        logger.info(f"✅ Successfully set {len(school_ids)} school(s) for {email}")
        
        return {
            "success": True,
            "user": user_with_schools,
            "message": f"Selected {len(school_ids)} school(s)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting schools: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/user-schools/{email}")
async def get_user_selected_schools(email: str):
    """
    Get all schools selected by a user
    """
    try:
        email = email.strip().lower()
        schools = get_user_schools(email)
        
        return {
            "success": True,
            "schools": schools,
            "count": len(schools)
        }
    except Exception as e:
        logger.error(f"Error getting user schools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/reset-database")
@app.get("/api/auth/reset-database")  # Allow GET for browser access
async def reset_database_endpoint():
    """
    Reset the entire database - drops all tables and recreates them.
    WARNING: This will delete ALL data including users and school selections!
    Use only for development/testing purposes.
    """
    try:
        logger.warning("="*80)
        logger.warning("🚨 DATABASE RESET REQUESTED")
        logger.warning("="*80)
        
        reset_database()
        
        logger.info("="*80)
        logger.info("✅ DATABASE RESET SUCCESSFUL")
        logger.info("="*80)
        
        return {
            "success": True,
            "message": "Database reset successfully. All data has been cleared and tables recreated.",
            "warning": "All users and school selections have been deleted."
        }
    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")


# ============================================================
# BOOKMARKS ENDPOINTS
# ============================================================

@app.post("/api/bookmarks/add")
async def add_user_bookmark(request: BookmarkRequest):
    """
    Add a bookmark for a user
    """
    try:
        email = request.email.strip().lower()
        
        add_bookmark(
            email=email,
            bookmark_id=request.bookmark_id,
            message_type=request.message_type,
            message_content=request.message_content,
            message_context=request.message_context,
            message_source=request.message_source,
            message_index=request.message_index
        )
        
        return {
            "success": True,
            "message": "Bookmark added successfully"
        }
    except Exception as e:
        logger.error(f"Error adding bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bookmarks/remove")
async def remove_user_bookmark(request: RemoveBookmarkRequest):
    """
    Remove a bookmark for a user
    """
    try:
        email = request.email.strip().lower()
        remove_bookmark(email, request.bookmark_id)
        
        return {
            "success": True,
            "message": "Bookmark removed successfully"
        }
    except Exception as e:
        logger.error(f"Error removing bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bookmarks/{email}")
async def get_bookmarks(email: str):
    """
    Get all bookmarks for a user
    """
    try:
        email = email.strip().lower()
        bookmarks = get_user_bookmarks(email)
        
        return {
            "success": True,
            "bookmarks": bookmarks,
            "count": len(bookmarks)
        }
    except Exception as e:
        logger.error(f"Error getting bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# USER PREFERENCES ENDPOINTS
# ============================================================

@app.post("/api/preferences/set")
async def set_preference(request: PreferenceRequest):
    """
    Set or update a user preference
    """
    try:
        email = request.email.strip().lower()
        set_user_preference(email, request.preference_key, request.preference_value)
        
        return {
            "success": True,
            "message": "Preference set successfully"
        }
    except Exception as e:
        logger.error(f"Error setting preference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preferences/{email}")
async def get_preferences(email: str):
    """
    Get all preferences for a user
    """
    try:
        email = email.strip().lower()
        preferences = get_all_user_preferences(email)
        
        return {
            "success": True,
            "preferences": preferences
        }
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preferences/{email}/{preference_key}")
async def get_preference(email: str, preference_key: str):
    """
    Get a specific preference for a user
    """
    try:
        email = email.strip().lower()
        value = get_user_preference(email, preference_key)
        
        return {
            "success": True,
            "preference_key": preference_key,
            "preference_value": value
        }
    except Exception as e:
        logger.error(f"Error getting preference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schools")
async def get_schools():
    """Get list of available schools"""
    schools = [
        {"id": 1, "district": "Round Rock", "email_suffix": "roundrockisd.org"},
        {"id": 2, "district": "Austin", "email_suffix": "austinisd.org"},
        {"id": 3, "district": "Leander", "email_suffix": "leanderisd.org"},
        {"id": 4, "district": "Pflugerville", "email_suffix": "pfisd.net"},
        {"id": 5, "district": "Georgetown", "email_suffix": "georgetownisd.org"},
        {"id": 6, "district": "Hutto", "email_suffix": "hutto.txed.net"},
        {"id": 7, "district": "Manor", "email_suffix": "manorisd.net"},
        {"id": 8, "district": "Lake Travis", "email_suffix": "ltisdschools.org"}
    ]
    return {"schools": schools}


# ============================================================
# RAGAS EVALUATION ENDPOINTS
# ============================================================

class EvaluationRequest(BaseModel):
    """Request model for RAGAS evaluation"""
    clear_buffer: bool = True  # Whether to clear the buffer after evaluation
    evaluation_name: str = "multi_agent_evaluation"

@app.post("/evaluation/run")
async def run_evaluation(request: EvaluationRequest):
    """
    Run RAGAS evaluation on collected multi-agent responses.
    
    Evaluates the actual queries and responses from /multi-agent-query endpoint
    that have been stored in the evaluation buffer.
    """
    try:
        from app.evaluation.ragas_evaluator import RAGASEvaluator
        
        global evaluation_buffer
        
        logger.info("="*80)
        logger.info(f"🔬 RAGAS Evaluation: {request.evaluation_name}")
        logger.info("="*80)
        
        # Check if we have any queries to evaluate
        if not evaluation_buffer or len(evaluation_buffer) == 0:
            logger.warning("⚠️ No queries in evaluation buffer")
            return {
                "status": "error",
                "message": "No queries to evaluate. Please run some queries through /multi-agent-query first.",
                "queries_needed": 10,
                "queries_collected": 0
            }
        
        logger.info(f"� Evaluating {len(evaluation_buffer)} queries from buffer")
        
        # Prepare queries and responses for evaluation
        queries_and_responses = []
        for item in evaluation_buffer:
            queries_and_responses.append({
                "user_input": item["user_input"],
                "response": item["response"],
                "retrieved_contexts": item.get("retrieved_contexts", [])
            })
        
        # Evaluate responses
        evaluator = RAGASEvaluator()
        logger.info("📈 Running RAGAS evaluation...")
        results = evaluator.evaluate_responses(
            queries_and_responses=queries_and_responses,
            evaluation_name=request.evaluation_name
        )
        
        # Add buffer info to results
        results["queries_evaluated"] = len(evaluation_buffer)
        results["buffer_cleared"] = request.clear_buffer
        
        # Clear buffer if requested
        if request.clear_buffer:
            logger.info("🧹 Clearing evaluation buffer")
            evaluation_buffer.clear()
        
        logger.info("="*80)
        logger.info("✅ Evaluation Complete")
        logger.info("="*80)
        
        return {
            "status": "success",
            "evaluation_name": request.evaluation_name,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/buffer")
async def get_evaluation_buffer_status():
    """Get the current status of the evaluation buffer."""
    global evaluation_buffer
    
    return {
        "queries_collected": len(evaluation_buffer),
        "buffer_sample": evaluation_buffer[-5:] if len(evaluation_buffer) > 0 else [],
        "message": f"Collected {len(evaluation_buffer)} queries. Need at least 1 query to run evaluation."
    }


@app.post("/evaluation/buffer/clear")
async def clear_evaluation_buffer():
    """Clear the evaluation buffer."""
    global evaluation_buffer
    count = len(evaluation_buffer)
    evaluation_buffer.clear()
    
    logger.info(f"🧹 Cleared evaluation buffer ({count} queries removed)")
    
    return {
        "status": "success",
        "message": f"Cleared {count} queries from evaluation buffer"
    }


@app.get("/retrieval-methods")
async def get_retrieval_methods():
    """Get all available retrieval methods"""
    return {
        "methods": RETRIEVAL_METHODS,
        "active": active_retrieval_method
    }

class RetrievalMethodRequest(BaseModel):
    method: str

@app.post("/retrieval-method")
async def set_retrieval_method(request: RetrievalMethodRequest):
    """Switch the active retrieval method"""
    global active_retrieval_method, retriever, generator_chain
    global original_retriever, naive_retriever, original_chain, naive_chain
    
    if request.method not in RETRIEVAL_METHODS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid method. Choose from: {list(RETRIEVAL_METHODS.keys())}"
        )
    
    # Check if retrievers are initialized
    if not original_retriever or not naive_retriever:
        logger.info("⚙️ Retrievers not initialized, setting up...")
        setup_rag_pipeline()
    
    # Switch the active method
    old_method = active_retrieval_method
    active_retrieval_method = request.method
    
    if active_retrieval_method == "original":
        retriever = original_retriever
        generator_chain = original_chain
    else:
        retriever = naive_retriever
        generator_chain = naive_chain
    
    logger.info("="*80)
    logger.info(f"🔄 RETRIEVAL METHOD SWITCHED")
    logger.info(f"   From: {RETRIEVAL_METHODS[old_method]}")
    logger.info(f"   To: {RETRIEVAL_METHODS[active_retrieval_method]}")
    logger.info("="*80)
    
    return {
        "status": "success",
        "previous_method": old_method,
        "active_method": active_retrieval_method,
        "description": RETRIEVAL_METHODS[active_retrieval_method]
    }

@app.get("/retrieval-method")
async def get_active_retrieval_method():
    """Get the currently active retrieval method"""
    return {
        "active_method": active_retrieval_method,
        "description": RETRIEVAL_METHODS[active_retrieval_method]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "rag_initialized": retriever is not None}


@app.get("/test-gmail", response_class=HTMLResponse)
async def test_gmail_page():
    """Serve Gmail integration test page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gmail Integration Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .section {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success { border-left: 4px solid #28a745; }
        .error { border-left: 4px solid #dc3545; }
        .info { border-left: 4px solid #17a2b8; }
        .warning { border-left: 4px solid #ffc107; }
        button {
            padding: 10px 20px;
            margin: 5px;
            cursor: pointer;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            font-size: 14px;
        }
        button:hover { background: #0056b3; }
        input {
            padding: 10px;
            width: 100%;
            max-width: 400px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            overflow-x: auto;
            border-radius: 4px;
            font-size: 12px;
        }
        h1 { color: #333; }
        h2 { color: #555; font-size: 18px; margin-top: 0; }
        .result-box {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            min-height: 50px;
        }
    </style>
</head>
<body>
    <h1>🔧 Gmail Integration Test</h1>
    
    <div class="section info">
        <h2>📋 Step 1: Check Gmail Status in Database</h2>
        <input type="email" id="userEmail" placeholder="Enter your email" value="vipinvijayan23@gmail.com">
        <br>
        <button onclick="checkDatabase()">Check Database</button>
        <div id="dbResult" class="result-box"></div>
    </div>
    
    <div class="section info">
        <h2>🌐 Step 2: Test WebSocket Query</h2>
        <input type="text" id="query" placeholder="Enter query" value="school programs">
        <br>
        <button onclick="testWebSocket()">Send WebSocket Query</button>
        <button onclick="clearMessages()">Clear Messages</button>
        <div id="wsResult" class="result-box"></div>
    </div>
    
    <div class="section info">
        <h2>📊 Step 3: Database Tables</h2>
        <button onclick="loadTables()">Load Database Tables</button>
        <div id="tablesResult" class="result-box"></div>
    </div>
    
    <div class="section info">
        <h2>📝 Step 4: Backend Logs</h2>
        <p><em>Check terminal/logs for backend activity</em></p>
        <pre>tail -f backend/nohup.out</pre>
    </div>

    <script>
        const API_URL = window.location.origin;
        
        async function checkDatabase() {
            const result = document.getElementById('dbResult');
            const email = document.getElementById('userEmail').value;
            result.innerHTML = '<p>⏳ Checking database...</p>';
            
            try {
                const response = await fetch(API_URL + '/api/auth/gmail/status?email=' + encodeURIComponent(email));
                const data = await response.json();
                
                const parent = result.closest('.section');
                parent.className = data.connected ? 'section success' : 'section error';
                
                result.innerHTML = '<strong>Result:</strong><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                
                if (!data.connected) {
                    result.innerHTML += '<p style="color: #dc3545;">❌ Gmail not connected! Please sign in with Gmail OAuth first.</p>';
                }
            } catch (error) {
                result.innerHTML = '<p style="color: #dc3545;">❌ Error: ' + error.message + '</p>';
                result.closest('.section').className = 'section error';
            }
        }
        
        function clearMessages() {
            document.getElementById('wsResult').innerHTML = '';
            document.getElementById('wsResult').closest('.section').className = 'section info';
        }
        
        function testWebSocket() {
            const email = document.getElementById('userEmail').value;
            const query = document.getElementById('query').value;
            const result = document.getElementById('wsResult');
            
            result.innerHTML = '<p>⏳ Connecting to WebSocket...</p>';
            
            const ws = new WebSocket('ws://' + window.location.host + '/ws/multi-agent-stream');
            
            let gmailFound = false;
            
            ws.onopen = () => {
                result.innerHTML = '<p>✅ Connected! Sending query...</p>';
                
                const message = {
                    question: query,
                    user_email: email,
                    email_suffixes: null,
                    school_districts: null
                };
                
                console.log('📤 Sending:', message);
                result.innerHTML += '<pre>Sent: ' + JSON.stringify(message, null, 2) + '</pre>';
                
                ws.send(JSON.stringify(message));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('📥 Received:', data);
                
                if (data.type === 'update' && data.agent === 'Gmail') {
                    gmailFound = true;
                    const content = data.content;
                    
                    result.innerHTML += '<div style="margin: 15px 0; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">';
                    result.innerHTML += '<strong>📧 Gmail Agent Response:</strong><br>';
                    result.innerHTML += content;
                    result.innerHTML += '</div>';
                    
                    if (content.includes('unable to access') || content.includes('not connected')) {
                        result.closest('.section').className = 'section error';
                        result.innerHTML += '<p style="color: #dc3545; font-weight: bold;">❌ Gmail search FAILED!</p>';
                    } else if (content.includes('Found') && content.includes('email')) {
                        result.closest('.section').className = 'section success';
                        result.innerHTML += '<p style="color: #28a745; font-weight: bold;">✅ Gmail search SUCCESS!</p>';
                    }
                }
                
                if (data.type === 'final') {
                    result.innerHTML += '<p style="margin-top: 15px;"><strong>✅ Query complete!</strong></p>';
                    
                    if (!gmailFound) {
                        result.innerHTML += '<p style="color: #dc3545;">⚠️ Warning: No Gmail agent response received!</p>';
                        result.closest('.section').className = 'section warning';
                    }
                    
                    ws.close();
                }
            };
            
            ws.onerror = (error) => {
                result.innerHTML += '<p style="color: #dc3545;">❌ WebSocket Error</p>';
                result.closest('.section').className = 'section error';
            };
            
            ws.onclose = () => {
                result.innerHTML += '<p><em>Connection closed</em></p>';
            };
        }
        
        async function loadTables() {
            const result = document.getElementById('tablesResult');
            result.innerHTML = '<p>⏳ Loading database tables...</p>';
            
            try {
                const response = await fetch(API_URL + '/api/debug/tables');
                const data = await response.json();
                
                result.closest('.section').className = 'section success';
                
                let html = '<h3>Database Tables</h3>';
                
                for (const [tableName, tableData] of Object.entries(data.tables)) {
                    html += '<div style="margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">';
                    html += '<div style="background: #007bff; color: white; padding: 10px; font-weight: bold;">';
                    html += '📋 ' + tableName + ' (' + tableData.count + ' rows)';
                    html += '</div>';
                    
                    if (tableData.data && tableData.data.length > 0) {
                        html += '<div style="overflow-x: auto;">';
                        html += '<table style="width: 100%; border-collapse: collapse;">';
                        
                        // Header
                        html += '<thead><tr style="background: #f8f9fa;">';
                        for (const col of tableData.columns) {
                            html += '<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">' + col + '</th>';
                        }
                        html += '</tr></thead>';
                        
                        // Data rows
                        html += '<tbody>';
                        for (const row of tableData.data) {
                            html += '<tr>';
                            for (const col of tableData.columns) {
                                let value = row[col];
                                
                                // Truncate long values
                                if (typeof value === 'string' && value.length > 100) {
                                    value = value.substring(0, 100) + '...';
                                }
                                
                                // Highlight NULL values
                                if (value === null || value === undefined) {
                                    value = '<em style="color: #999;">NULL</em>';
                                }
                                
                                html += '<td style="padding: 8px; border: 1px solid #ddd;">' + value + '</td>';
                            }
                            html += '</tr>';
                        }
                        html += '</tbody>';
                        html += '</table>';
                        html += '</div>';
                    } else {
                        html += '<div style="padding: 20px; text-align: center; color: #999;">No data</div>';
                    }
                    
                    html += '</div>';
                }
                
                result.innerHTML = html;
            } catch (error) {
                result.innerHTML = '<p style="color: #dc3545;">❌ Error: ' + error.message + '</p>';
                result.closest('.section').className = 'section error';
            }
        }
        
        // Auto-load on page load
        window.onload = () => {
            console.log('Test page loaded. API URL:', API_URL);
        };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/debug/tables")
async def get_debug_tables():
    """
    Get all tables and their data from the database for debugging.
    Returns table names, row counts, column names, and data.
    """
    import sqlite3
    
    try:
        db_path = os.path.join(DB_DIR, "school_assistant.db")
        if not os.path.exists(db_path):
            return {"error": f"Database file not found at {db_path}", "tables": {}}
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [row[0] for row in cursor.fetchall()]
        
        tables_data = {}
        
        for table_name in table_names:
            # Skip SQLite internal tables
            if table_name.startswith('sqlite_'):
                continue
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Get all data (limit to 100 rows for safety)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
            rows = cursor.fetchall()
            
            # Convert rows to list of dicts
            data = []
            for row in rows:
                row_dict = {}
                for col in columns:
                    value = row[col]
                    # Truncate long values in the response itself
                    if value and isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    row_dict[col] = value
                data.append(row_dict)
            
            tables_data[table_name] = {
                "count": count,
                "columns": columns,
                "data": data
            }
        
        conn.close()
        
        return {"tables": tables_data}
        
    except Exception as e:
        print(f"Error reading database: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "tables": {}}


@app.get("/events")
async def get_events():
    """Get all events from the data directory"""
    try:
        events = []
        
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.txt'):
                filepath = os.path.join(DATA_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Parse the text file to extract key information
                    lines = content.split('\n')
                    
                    # First line is typically the event/program name
                    event_name = lines[0].strip() if lines else filename.replace('.txt', '').replace('_', ' ').title()
                    
                    # Extract organization, category, and age range from second paragraph
                    organization = ""
                    category = ""
                    target_audience = ""
                    description = ""
                    cost = ""
                    date = ""
                    
                    # Parse the content for key information
                    for i, line in enumerate(lines):
                        line_lower = line.lower().strip()
                        
                        # Extract organization (usually in first few lines)
                        if 'organized by' in line_lower or 'organization:' in line_lower:
                            parts = line.split('organized by')
                            if len(parts) > 1:
                                organization = parts[1].split('.')[0].strip()
                        
                        # Extract category
                        if 'category' in line_lower and not category:
                            if 'falls under the' in line_lower:
                                parts = line.split('falls under the')
                                if len(parts) > 1:
                                    category = parts[1].split('category')[0].strip()
                        
                        # Extract age range
                        if 'age range' in line_lower or 'ages ' in line_lower or 'grades' in line_lower:
                            if 'designed for' in line_lower or 'age range' in line_lower:
                                # Extract the age/grade info
                                if 'ages' in line_lower:
                                    import re
                                    age_match = re.search(r'ages?\s+[\d\-\+\s]+', line_lower)
                                    if age_match:
                                        target_audience = age_match.group(0).title()
                                if 'grades' in line_lower:
                                    import re
                                    grade_match = re.search(r'grades?\s+[k\d\-\+\s]+', line_lower, re.IGNORECASE)
                                    if grade_match:
                                        target_audience = grade_match.group(0).title()
                        
                        # Extract pricing
                        if 'pricing information:' in line_lower or 'cost:' in line_lower:
                            # Look for price in next few lines
                            for j in range(i, min(i+10, len(lines))):
                                price_line = lines[j]
                                if '$' in price_line:
                                    # Extract first price found
                                    import re
                                    price_match = re.search(r'\$\d+(?:\.\d{2})?(?:\s*(?:per|/)\s*\w+)?', price_line)
                                    if price_match and not cost:
                                        cost = price_match.group(0)
                                        break
                        
                        # Extract dates (look for date patterns)
                        if any(month in line_lower for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                                   'july', 'august', 'september', 'october', 'november', 'december']) and not date:
                            date = line.strip()[:100]  # Limit length
                    
                    # Create description from first paragraph after title
                    if len(lines) > 1:
                        # Get first substantial paragraph
                        for line in lines[1:]:
                            if len(line.strip()) > 50:
                                description = line.strip()[:200]  # Limit to 200 chars
                                break
                    
                    event = {
                        "id": filename.replace('.txt', ''),
                        "name": event_name,
                        "organization": organization,
                        "description": description,
                        "target_audience": target_audience,
                        "date": date,
                        "cost": cost,
                        "type": category if category else "Event",
                        "filename": filename
                    }
                    
                    events.append(event)
        
        logger.info(f"✅ Retrieved {len(events)} events")
        return {"events": events, "count": len(events)}
        
    except Exception as e:
        logger.error(f"❌ Error retrieving events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving events: {str(e)}")



@app.post("/multi-agent-query")
async def multi_agent_query_events(request: QueryRequest):
    """Process user query using multi-agent system with routing"""
    from datetime import datetime
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info(f"📥 /multi-agent-query ENDPOINT")
    logger.info(f"Query: {request.question}")
    logger.info(f"Email Suffix (single): {request.email_suffix}")
    logger.info(f"Email Suffixes (multiple): {request.email_suffixes}")
    
    try:
        # Set email suffixes for Gmail tool - support both single and multiple schools
        from app.tools.gmail_tool import get_gmail_client
        gmail_client = get_gmail_client()
        
        if request.email_suffixes and len(request.email_suffixes) > 0:
            # Multi-school mode
            gmail_client.set_email_suffixes(request.email_suffixes)
            logger.info(f"✉️ Gmail search will filter by {len(request.email_suffixes)} school(s): {', '.join(['@' + s for s in request.email_suffixes])}")
        elif request.email_suffix:
            # Legacy single school mode
            gmail_client.set_email_suffix(request.email_suffix)
            logger.info(f"✉️ Gmail search will filter by: @{request.email_suffix}")
        
        # Set school context for Tavily tool if school info is provided
        if request.school_districts and len(request.school_districts) > 0:
            # Multi-school mode
            from app.tools.tavily_tool import get_tavily_client
            tavily_client = get_tavily_client()
            # Use the first district as primary context (Tavily doesn't support multi-context yet)
            district_name = request.school_districts[0]
            tavily_client.set_school_context(district=district_name, email_suffix=request.email_suffixes[0] if request.email_suffixes else None)
            logger.info(f"🏫 Tavily search context set to: {district_name} (+ {len(request.school_districts)-1} more)")
        elif request.school_district or request.email_suffix:
            # Legacy single school mode
            from app.tools.tavily_tool import get_tavily_client
            tavily_client = get_tavily_client()
            
            # Use provided district name, or create one from email suffix
            district_name = request.school_district
            if not district_name and request.email_suffix:
                district_name = request.email_suffix.split('.')[0].replace('isd', ' ISD').replace('txed', '').title()
            
            tavily_client.set_school_context(district=district_name, email_suffix=request.email_suffix)
            logger.info(f"🏫 Tavily search context set to: {district_name}")
        
        # Initialize multi-agents if not already done
        global multi_agents, evaluation_buffer
        if not multi_agents:
            logger.info("⚙️ Multi-Agent system not initialized, setting up...")
            print("\n🤖 Initializing Multi-Agent System...")
            multi_agents = create_school_events_agents()
            print("✅ Multi-Agent System initialized")
            logger.info("✅ Multi-Agent system ready")
        
        # Use the agent graph to process the query
        logger.info("🚀 Routing query through multi-agent system...")
        result = query_with_agent(request.question)
        
        # Extract the response from messages
        if "messages" in result and len(result["messages"]) > 0:
            # Get the last message which should be the agent's response
            last_message = result["messages"][-1]
            response_text = last_message.content
            agent_name = getattr(last_message, 'name', 'Unknown')
            
            # Map agent name directly to source
            source_map = {
                "LocalEvents": "Local Database",
                "GmailAgent": "Gmail",
                "WebSearch": "Web Search"
            }
            source = source_map.get(agent_name, "Unknown")
            
            # Remove source tag from response if LLM included it
            clean_response = response_text
            for tag in ["[Source: Local Database]", "[Source: Gmail]", "[Source: Web Search]"]:
                if tag in clean_response:
                    clean_response = clean_response.replace(tag, "").strip()
                    break
            
            # Store query and response for evaluation
            contexts = [msg.content for msg in result.get("messages", []) if hasattr(msg, 'content') and msg.content]
            query_data = {
                "user_input": request.question,
                "response": clean_response,
                "retrieved_contexts": contexts,
                "agent_used": agent_name,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
            evaluation_buffer.append(query_data)
            logger.info(f"📊 Stored query in evaluation buffer (total: {len(evaluation_buffer)})")
            
            # Trigger automatic evaluation in background
            evaluation_result = None
            try:
                logger.info("🔬 Starting automatic RAGAS evaluation...")
                from app.evaluation.ragas_evaluator import RAGASEvaluator
                evaluator = RAGASEvaluator()
                
                # Evaluate just this single query
                evaluation_result = evaluator.evaluate_responses(
                    queries_and_responses=[query_data],
                    evaluation_name=f"auto_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logger.info(f"✅ Evaluation complete: Faithfulness={evaluation_result['metrics']['faithfulness']:.3f}, Relevancy={evaluation_result['metrics']['response_relevancy']:.3f}")
            except Exception as eval_error:
                logger.error(f"⚠️ Evaluation failed (non-blocking): {str(eval_error)}")
                evaluation_result = {
                    "error": str(eval_error),
                    "metrics": {
                        "faithfulness": 0,
                        "response_relevancy": 0
                    }
                }
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Multi-agent query completed in {duration:.2f}s")
            logger.info(f"Agent used: {agent_name}")
            logger.info(f"Source: {source}")
            logger.info(f"Messages exchanged: {len(result['messages'])}")
            logger.info(f"Response length: {len(clean_response)} characters")
            logger.info("="*80)
            
            return {
                "answer": clean_response,
                "agent_used": agent_name,
                "source": source,
                "response_time": round(duration, 2),
                "message_count": len(result["messages"]),
                "context": [clean_response[:200] + "..."] if len(clean_response) > 200 else [clean_response],
                "evaluation": {
                    "faithfulness": evaluation_result["metrics"]["faithfulness"] if evaluation_result else 0,
                    "response_relevancy": evaluation_result["metrics"]["response_relevancy"] if evaluation_result else 0,
                    "status": "completed" if evaluation_result and "error" not in evaluation_result else "failed"
                }
            }
        else:
            duration = (datetime.now() - start_time).total_seconds()
            logger.warning(f"⚠️ No response generated after {duration:.2f}s")
            logger.warning("="*80)
            return {
                "answer": "No response generated from agents",
                "agent_used": "None",
                "message_count": 0,
                "context": [],
                "evaluation": {
                    "faithfulness": 0,
                    "response_relevancy": 0,
                    "status": "no_response"
                }
            }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Multi-agent query failed after {duration:.2f}s: {str(e)}")
        logger.error("="*80)
        print(f"Error in multi_agent_query_events: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing multi-agent query: {str(e)}")


# ============================================================
# GMAIL OAUTH ENDPOINTS
# ============================================================

from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import google.auth.exceptions

# Gmail OAuth configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
REDIRECT_URI = 'http://localhost:8000/api/auth/gmail/callback'
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials', 'gmail_credentials.json')

def get_gmail_oauth_flow():
    """Get OAuth flow from file or environment variables"""
    # Try to use credentials file first
    if os.path.exists(CREDENTIALS_PATH):
        return Flow.from_client_secrets_file(
            CREDENTIALS_PATH,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
    
    # Fall back to environment variables
    client_id = os.getenv('GOOGLE_CLIENT_ID')
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError(
            "Gmail OAuth not configured. Please either:\n"
            "1. Create credentials/gmail_credentials.json file, OR\n"
            "2. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables"
        )
    
    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI]
        }
    }
    
    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

@app.get("/api/auth/gmail/authorize")
async def gmail_authorize(email: str):
    """Generate Gmail OAuth authorization URL"""
    try:
        # Create OAuth flow
        flow = get_gmail_oauth_flow()
        
        # Generate authorization URL with user email as state
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=email  # Pass user email as state to retrieve in callback
        )
        
        logger.info(f"📧 Generated Gmail OAuth URL for: {email}")
        return {
            "authorization_url": authorization_url,
            "state": state
        }
        
    except Exception as e:
        logger.error(f"Error generating OAuth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/gmail/callback")
async def gmail_callback(code: str, state: str):
    """Handle Gmail OAuth callback, create/login user, and save token"""
    try:
        # Exchange authorization code for tokens
        flow = get_gmail_oauth_flow()
        flow.fetch_token(code=code)
        
        # Get credentials
        creds = flow.credentials
        
        # Get Gmail email address from Google
        from googleapiclient.discovery import build
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        gmail_email = profile.get('emailAddress')
        
        # Create or get user with this Gmail email
        user = get_or_create_user(gmail_email)
        
        # Save token to database for this user
        token_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes,
            'expiry': creds.expiry.isoformat() if creds.expiry else None
        }
        
        save_user_gmail_token(gmail_email, json.dumps(token_data), gmail_email)
        
        logger.info(f"✅ Gmail OAuth login successful for: {gmail_email}")
        
        # Return HTML that closes popup and signals success to parent window
        return HTMLResponse(content=f"""
            <html>
                <head><title>Sign In Successful</title></head>
                <body>
                    <script>
                        // Store the Gmail email in a way the parent can access
                        window.opener.postMessage({{
                            type: 'gmail_oauth_success',
                            email: '{gmail_email}'
                        }}, '*');
                        window.close();
                    </script>
                    <div style="
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        font-family: Arial, sans-serif;
                        flex-direction: column;
                        gap: 20px;
                    ">
                        <h2>✅ Sign In Successful!</h2>
                        <p>You can close this window now.</p>
                        <p style="color: #666; font-size: 0.9em;">Redirecting...</p>
                    </div>
                </body>
            </html>
        """)
        
    except Exception as e:
        logger.error(f"Error in Gmail callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/gmail/disconnect")
async def gmail_disconnect(request: dict):
    """Disconnect user's Gmail account"""
    try:
        email = request.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email required")
        
        disconnect_user_gmail(email)
        logger.info(f"🔌 Gmail disconnected for: {email}")
        
        return {"status": "success", "message": "Gmail account disconnected"}
        
    except Exception as e:
        logger.error(f"Error disconnecting Gmail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/gmail/status")
async def gmail_status(email: str):
    """Check if user has Gmail connected"""
    try:
        token_data = get_user_gmail_token(email)
        
        if token_data:
            return {
                "connected": True,
                "gmail_email": token_data['gmail_email'],
                "connected_at": token_data['connected_at']
            }
        else:
            return {"connected": False}
            
    except Exception as e:
        logger.error(f"Error checking Gmail status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# WEBSOCKET ENDPOINTS
# ============================================================

@app.websocket("/ws/multi-agent-stream")
async def websocket_multi_agent_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming multi-agent responses in real-time"""
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")
    
    try:
        while True:
            # Receive the query from client
            data = await websocket.receive_json()
            logger.info(f"📦 Raw WebSocket data received: {data}")
            
            question = data.get("question", "")
            user_email = data.get("user_email")  # User email for per-user Gmail auth
            email_suffix = data.get("email_suffix")  # Legacy single school
            email_suffixes = data.get("email_suffixes")  # Multi-school support
            school_district = data.get("school_district")  # Legacy single school
            school_districts = data.get("school_districts")  # Multi-school support
            
            if not question:
                await websocket.send_json({
                    "type": "error",
                    "content": "No question provided"
                })
                continue
            
            logger.info(f"📥 WebSocket query received: {question}")
            logger.info(f"👤 User Email: {user_email}")
            
            # Warn if no user email (Gmail won't work)
            if not user_email:
                logger.warning("⚠️ No user_email received from frontend - Gmail search will not work!")
                logger.warning("⚠️ User may need to refresh page or re-login")
            
            logger.info(f"Email Suffix (single): {email_suffix}")
            logger.info(f"Email Suffixes (multiple): {email_suffixes}")
            
            # Set school context for Tavily tool if school info is provided
            if school_districts and len(school_districts) > 0:
                # Multi-school mode
                from app.tools.tavily_tool import get_tavily_client
                tavily_client = get_tavily_client()
                # Use the first district as primary context
                district_name = school_districts[0]
                tavily_client.set_school_context(district=district_name, email_suffix=email_suffixes[0] if email_suffixes else None)
                logger.info(f"🏫 Tavily search context set to: {district_name} (+ {len(school_districts)-1} more)")
            elif school_district or email_suffix:
                # Legacy single school mode
                from app.tools.tavily_tool import get_tavily_client
                tavily_client = get_tavily_client()
                
                # Use provided district name, or create one from email suffix
                district_name = school_district
                if not district_name and email_suffix:
                    district_name = email_suffix.split('.')[0].replace('isd', ' ISD').replace('txed', '').title()
                
                tavily_client.set_school_context(district=district_name, email_suffix=email_suffix)
                logger.info(f"🏫 Tavily search context set to: {district_name}")
            
            # Initialize multi-agents with current user's email for per-user Gmail auth
            # Note: We recreate agents each time to ensure correct per-user credentials
            logger.info(f"⚙️ Setting up Multi-Agent system for user: {user_email}")
            await websocket.send_json({
                "type": "status",
                "content": "Initializing search..."
            })
            multi_agents = create_school_events_agents(user_email=user_email)
            logger.info("✅ Multi-Agent system ready with user-specific credentials")
            
            # Set email suffixes for Gmail tool AFTER agents are created
            from app.tools.gmail_tool import get_gmail_client
            gmail_client = get_gmail_client()
            
            if email_suffixes and len(email_suffixes) > 0:
                # Multi-school mode
                gmail_client.set_email_suffixes(email_suffixes)
                logger.info(f"✉️ Gmail search will filter by {len(email_suffixes)} school(s): {', '.join(['@' + s for s in email_suffixes])}")
            elif email_suffix:
                # Legacy single school mode
                gmail_client.set_email_suffix(email_suffix)
                logger.info(f"✉️ Gmail search will filter by: @{email_suffix}")
            
            # Store final response for evaluation
            final_response = None
            final_agent_name = None
            
            # Define callback to send updates via WebSocket
            async def send_update(agent_name: str, content: str, is_final: bool, tool_name: str = None, duration: float = None):
                nonlocal final_response, final_agent_name
                try:
                    message = {
                        "type": "final" if is_final else "update",
                        "agent": agent_name,
                        "content": content
                    }
                    
                    if tool_name:
                        message["tool"] = tool_name
                    
                    if is_final and duration:
                        message["response_time"] = round(duration, 2)
                        final_response = content
                        final_agent_name = agent_name
                    
                    await websocket.send_json(message)
                    logger.info(f"📤 Sent {'final' if is_final else 'update'} from {agent_name}")
                except Exception as e:
                    logger.error(f"❌ Error sending WebSocket update: {e}")
            
            # Process query with streaming
            try:
                result = await query_with_agent_stream(question, send_update, agents=multi_agents)
                
                # Run evaluation after streaming completes
                if final_response and result:
                    try:
                        logger.info("🔬 Starting automatic RAGAS evaluation for WebSocket...")
                        
                        # Prepare query data for evaluation
                        # For evaluation, we need retrieved_contexts
                        # Use the final response as the context (simplified approach)
                        contexts = [final_response]
                        
                        # Try to extract more contexts from messages if available
                        try:
                            if isinstance(result, dict) and "messages" in result:
                                messages = result["messages"]
                                logger.info(f"📝 Found {len(messages)} messages in result")
                                for msg in messages:
                                    if hasattr(msg, 'content') and msg.content and msg.content != final_response:
                                        contexts.append(str(msg.content)[:1000])  # Limit context size
                        except Exception as ctx_error:
                            logger.warning(f"⚠️ Could not extract additional contexts: {ctx_error}")
                        
                        logger.info(f"📚 Using {len(contexts)} context(s) for evaluation")
                        
                        query_data = {
                            "user_input": question,
                            "response": final_response,
                            "retrieved_contexts": contexts,
                            "agent_used": final_agent_name,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        logger.info(f"📊 Query data prepared: question={question[:50]}..., response length={len(final_response)}, contexts={len(contexts)}")
                        
                        # Run RAGAS evaluation in a separate process to avoid uvloop conflicts
                        logger.info("🔧 Running RAGAS in separate process to avoid uvloop conflicts...")
                        import subprocess
                        import sys
                        import concurrent.futures
                        
                        # Prepare data for subprocess
                        eval_input = {
                            "query_data": query_data,
                            "evaluation_name": f"auto_eval_ws_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        }
                        
                        # Run evaluation in separate process with standard asyncio (not uvloop)
                        script_path = os.path.join(os.path.dirname(__file__), "run_ragas_standalone.py")
                        python_path = sys.executable
                        
                        def run_subprocess():
                            """Run RAGAS in subprocess (blocking call in thread)"""
                            result = subprocess.run(
                                [python_path, script_path, json.dumps(eval_input)],
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                            
                            if result.returncode != 0:
                                raise Exception(f"RAGAS subprocess failed: {result.stderr}")
                            
                            return json.loads(result.stdout)
                        
                        # Run in thread pool executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            evaluation_result = await loop.run_in_executor(executor, run_subprocess)
                        
                        # Send evaluation as a separate message
                        await websocket.send_json({
                            "type": "evaluation",
                            "evaluation": {
                                "faithfulness": evaluation_result["metrics"]["faithfulness"],
                                "response_relevancy": evaluation_result["metrics"]["response_relevancy"],
                                "status": "completed"
                            }
                        })
                        logger.info(f"✅ Evaluation sent: Faithfulness={evaluation_result['metrics']['faithfulness']:.3f}, Relevancy={evaluation_result['metrics']['response_relevancy']:.3f}")
                    except Exception as eval_error:
                        logger.error(f"⚠️ Evaluation failed (non-blocking): {str(eval_error)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        await websocket.send_json({
                            "type": "evaluation",
                            "evaluation": {
                                "faithfulness": 0,
                                "response_relevancy": 0,
                                "status": "failed",
                                "error": str(eval_error)
                            }
                        })
            except Exception as e:
                logger.error(f"❌ Error in streaming query: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error processing query: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket connection closed")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Server error: {str(e)}"
            })
        except:
            pass


def _run_ragas_evaluation_sync(active_method, orig_retriever, orig_chain, nv_retriever, nv_chain):
    """
    Synchronous function to run RAGAS evaluation in a separate thread.
    This avoids uvloop compatibility issues by running in a fresh event loop.
    
    Args:
        active_method: The active retrieval method ("original" or "naive")
        orig_retriever: The original retriever
        orig_chain: The original chain
        nv_retriever: The naive retriever
        nv_chain: The naive chain
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
        ContextPrecision
    )
    from ragas import RunConfig
    import pandas as pd
    
    logger.info("📚 Creating test dataset...")
    
    # Test questions covering all event types
    test_questions = [
        {
            "user_input": "What coding programs are available for kids?",
            "reference": "CodeWizardsHQ Logic Challenge for grades 3-12"
        },
        {
            "user_input": "What art classes does Cordovan Art School offer?",
            "reference": "Painting, Drawing, Anime, Clay and more for ages 4-Adult"
        },
        {
            "user_input": "Are there any music programs for children?",
            "reference": "National Children's Chorus with Junior and Senior divisions"
        },
        {
            "user_input": "What are the school holiday camp options?",
            "reference": "All-inclusive day camps with sports, STEM, martial arts"
        },
        {
            "user_input": "What sports programs are available?",
            "reference": "Texas Tomahawks lacrosse clinics for grades K-8"
        },
        {
            "user_input": "When are the art camp dates?",
            "reference": "October 2025 through January 2026"
        },
        {
            "user_input": "What age groups can participate in coding programs?",
            "reference": "Students in grades 3-12"
        },
        {
            "user_input": "Where is Cordovan Art School located?",
            "reference": "Cedar Park, Georgetown, NW Austin, Round Rock, SW Austin"
        }
    ]
    
    logger.info(f"✅ Created {len(test_questions)} test questions")
    
    # Run queries through RAG pipeline
    logger.info("🚀 Running queries through RAG pipeline...")
    eval_data = []
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["user_input"]
        reference = test_case["reference"]
        
        logger.info(f"  {i}/{len(test_questions)}: {question[:50]}...")
        
        # Query the RAG using the active method (passed as parameter)
        if active_method == "original":
            # Original method: manually retrieve and pass to chain
            retrieved_docs = orig_retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            response = orig_chain.invoke({
                "context": context,
                "question": question
            })
        else:  # naive
            # Naive method: LCEL chain handles retrieval internally
            result = nv_chain.invoke({"question": question})
            response = result["response"].content  # Extract content from AIMessage
            retrieved_docs = result["context"]  # This is the list of documents
        
        eval_data.append({
            "user_input": question,
            "response": response,
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
            "reference": reference
        })
    
    logger.info("✅ All queries completed")
    
    # Convert to RAGAS EvaluationDataset
    logger.info("📊 Preparing RAGAS evaluation dataset...")
    df = pd.DataFrame(eval_data)
    evaluation_dataset = EvaluationDataset.from_pandas(df)
    
    # Setup evaluator LLM
    logger.info("🤖 Setting up evaluator LLM (gpt-4o-mini)...")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    
    # Run RAGAS evaluation
    logger.info("🔬 Running RAGAS evaluation...")
    logger.info("   Metrics: Faithfulness, Response Relevancy, Context Precision, Context Recall")
    
    custom_run_config = RunConfig(timeout=360)
    
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision()
        ],
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    
    # Extract metrics
    result_df = result.to_pandas()
    
    # Debug: Log available columns
    logger.info(f"📋 Available RAGAS columns: {list(result_df.columns)}")
    logger.info(f"📊 Sample row:\n{result_df.head(1)}")
    
    # Extract metrics with actual column names
    available_cols = list(result_df.columns)
    metrics = {}
    
    # Map expected names to actual column names
    if "faithfulness" in available_cols:
        metrics["faithfulness"] = float(result_df["faithfulness"].mean())
    if "answer_relevancy" in available_cols:
        metrics["answer_relevancy"] = float(result_df["answer_relevancy"].mean())
    if "response_relevancy" in available_cols:
        metrics["response_relevancy"] = float(result_df["response_relevancy"].mean())
    if "context_precision" in available_cols:
        metrics["context_precision"] = float(result_df["context_precision"].mean())
    if "context_recall" in available_cols:
        metrics["context_recall"] = float(result_df["context_recall"].mean())
    
    # Save results to generated_results directory
    os.makedirs(GENERATED_RESULTS_DIR, exist_ok=True)
    
    results_csv_path = os.path.join(GENERATED_RESULTS_DIR, "ragas_evaluation_results.csv")
    summary_json_path = os.path.join(GENERATED_RESULTS_DIR, "ragas_evaluation_summary.json")
    
    result_df.to_csv(results_csv_path, index=False)
    
    with open(summary_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("="*80)
    logger.info("📊 RAGAS EVALUATION RESULTS")
    logger.info("="*80)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name:.<25} {metric_value:.4f}")
    logger.info("="*80)
    logger.info(f"💾 Results saved to: {results_csv_path}")
    logger.info(f"💾 Summary saved to: {summary_json_path}")
    logger.info("="*80)
    
    return {
        "status": "success",
        "metrics": metrics,
        "test_questions_count": len(test_questions),
        "files_generated": [
            results_csv_path,
            summary_json_path
        ],
        "message": "RAGAS evaluation completed successfully"
    }


@app.post("/evaluate-ragas")
async def evaluate_rag_with_ragas():
    """
    Evaluate the RAG pipeline using RAGAS framework
    
    This endpoint runs a comprehensive evaluation of the RAG pipeline using RAGAS metrics:
    - Faithfulness: Factual consistency with retrieved context
    - Response Relevancy: Relevance to the question
    - Context Precision: Precision of retrieved contexts
    - Context Recall: Coverage of required context
    """
    logger.info("="*80)
    logger.info("🔬 RAGAS EVALUATION ENDPOINT")
    logger.info("="*80)
    
    try:
        # Ensure RAG is initialized
        global retriever, generator_chain
        if retriever is None or generator_chain is None:
            logger.info("⚙️ RAG not initialized, setting up...")
            setup_rag_pipeline()
            logger.info("✅ RAG pipeline ready")
        
        # Run RAGAS evaluation in a separate thread to avoid uvloop compatibility issues
        # RAGAS tries to patch the event loop, but uvloop doesn't support it
        logger.info("🔧 Running RAGAS in separate thread to avoid uvloop conflicts...")
        import concurrent.futures
        
        # Pass all necessary data to the thread
        global active_retrieval_method, original_retriever, naive_retriever, original_chain, naive_chain
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                _run_ragas_evaluation_sync,
                active_retrieval_method,
                original_retriever,
                original_chain,
                naive_retriever,
                naive_chain
            )
            result = future.result()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ RAGAS evaluation failed: {str(e)}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error running RAGAS evaluation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")