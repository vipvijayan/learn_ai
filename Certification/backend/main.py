from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Directory paths - use absolute paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
GENERATED_RESULTS_DIR = os.path.join(SCRIPT_DIR, "generated_results")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("="*80)
logger.info("🚀 Starting School Events RAG API")
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
from app.agents.multi_agent_system import create_school_events_agents, query_with_agent

# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================

# CORS configuration
FRONTEND_URL = "http://localhost:3000"

# Retrieval method configuration
RETRIEVAL_METHODS = {
    "original": "Original RAG (k=4, query expansion)",
    "naive": "Naive Retrieval (k=10, LCEL chain from Advanced Retrieval)"
}

# ============================================================
# GLOBAL VARIABLES
# ============================================================

# RAG components
vector_store = None
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

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(title="School Events RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Log available endpoints and capabilities on startup"""
    logger.info("📋 AVAILABLE ENDPOINTS:")
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

# ============================================================
# PYDANTIC MODELS
# ============================================================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: List[str]

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
            tavily_tool = TavilySearchResults(max_results=3)
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

@app.post("/query", response_model=QueryResponse)
async def query_events(request: QueryRequest):
    """Process user query using the active retrieval method (Original or Naive)"""
    from datetime import datetime
    start_time = datetime.now()
    
    # Declare global variables FIRST
    global retriever, generator_chain, active_retrieval_method
    global original_retriever, naive_retriever, original_chain, naive_chain
    
    logger.info("="*80)
    logger.info(f"📥 /query ENDPOINT")
    logger.info(f"Query: {request.question}")
    logger.info(f"Active Method: {active_retrieval_method}")
    
    try:
        # Initialize RAG pipeline if not already done
        if not retriever or not generator_chain:
            logger.info("⚙️ RAG pipeline not initialized, setting up...")
            setup_rag_pipeline()
        
        if not retriever or not generator_chain:
            logger.error("❌ RAG pipeline initialization failed")
            raise HTTPException(status_code=500, detail="RAG pipeline initialization failed")
        
        # Route to the appropriate retrieval method
        if active_retrieval_method == "original":
            # ORIGINAL METHOD: k=4, simple chain
            logger.info(f"🔍 Using Original Retrieval (k=4)")
            retrieved_docs = original_retriever.invoke(request.question)
            
            logger.info(f"📄 Retrieved {len(retrieved_docs)} documents")
            
            # Log source files
            sources = [doc.metadata.get('source', 'unknown') for doc in retrieved_docs]
            unique_sources = list(set(sources))
            logger.info(f"📚 Sources used: {', '.join(unique_sources)}")
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Generate response
            logger.info(f"🤖 Using Original Chain (simple prompt | llm | parser)")
            response_text = original_chain.invoke({
                "question": request.question,
                "context": context
            })
            
            # Extract context snippets for display
            context_snippets = [doc.page_content[:200] + "..." for doc in retrieved_docs[:3]]
            
        else:  # naive method
            # NAIVE METHOD: k=10, LCEL chain
            logger.info(f"🔍 Using Naive Retrieval Chain (LCEL pattern, k=10)")
            result = naive_chain.invoke({"question": request.question})
            
            # Extract response and documents
            response_text = result["response"].content
            retrieved_docs = result["context"]
            
            logger.info(f"📄 Retrieved {len(retrieved_docs)} documents")
            
            # Log source files
            sources = [doc.metadata.get('source', 'unknown') for doc in retrieved_docs]
            unique_sources = list(set(sources))
            logger.info(f"📚 Sources used: {', '.join(unique_sources)}")
            
            # Extract context snippets for display
            context_snippets = [doc.page_content[:200] + "..." for doc in retrieved_docs[:3]]
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Query completed in {duration:.2f}s")
        logger.info(f"Response length: {len(response_text)} characters")
        logger.info(f"📊 Method used: {RETRIEVAL_METHODS[active_retrieval_method]}")
        logger.info("="*80)
        
        return QueryResponse(
            answer=response_text,
            context=context_snippets
        )
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Query failed after {duration:.2f}s: {str(e)}")
        logger.error("="*80)
        print(f"Error in query_events: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

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
    return {"status": "healthy", "rag_initialized": vector_store is not None}

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

@app.post("/agent-query")
async def agent_query_events(request: QueryRequest):
    """Process user query using agent tools (for demonstration/testing)"""
    from datetime import datetime
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info(f"📥 /agent-query ENDPOINT")
    logger.info(f"Query: {request.question}")
    
    try:
        # Initialize agent tools if not already done
        global agent_tools, school_events_tool
        if not agent_tools:
            logger.info("⚙️ Agent tools not initialized, setting up...")
            setup_agent_tools()
        
        if not school_events_tool:
            logger.error("❌ Agent tools initialization failed")
            raise HTTPException(status_code=500, detail="Agent tools initialization failed")
        
        # Use the school events tool directly
        logger.info(f"🔧 Using tool: {school_events_tool.name}")
        tool_result = school_events_tool._run(request.question)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Agent query completed in {duration:.2f}s")
        logger.info(f"Result length: {len(tool_result)} characters")
        logger.info("="*80)
        
        # Return structured response
        return {
            "answer": tool_result,
            "tool_used": school_events_tool.name,
            "context": [tool_result[:200] + "..."] if len(tool_result) > 200 else [tool_result]
        }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Agent query failed after {duration:.2f}s: {str(e)}")
        logger.error("="*80)
        print(f"Error in agent_query_events: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing agent query: {str(e)}")

@app.get("/tools")
async def list_tools():
    """List available agent tools"""
    try:
        global agent_tools
        if not agent_tools:
            setup_agent_tools()
        
        tools_info = []
        for tool in agent_tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
            })
        
        return {
            "tools": tools_info,
            "count": len(tools_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

@app.post("/multi-agent-query")
async def multi_agent_query_events(request: QueryRequest):
    """Process user query using multi-agent system with routing"""
    from datetime import datetime
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info(f"📥 /multi-agent-query ENDPOINT")
    logger.info(f"Query: {request.question}")
    
    try:
        # Initialize multi-agents if not already done
        global multi_agents
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
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Multi-agent query completed in {duration:.2f}s")
            logger.info(f"Agent used: {agent_name}")
            logger.info(f"Messages exchanged: {len(result['messages'])}")
            logger.info(f"Response length: {len(response_text)} characters")
            logger.info("="*80)
            
            return {
                "answer": response_text,
                "agent_used": agent_name,
                "message_count": len(result["messages"]),
                "context": [response_text[:200] + "..."] if len(response_text) > 200 else [response_text]
            }
        else:
            duration = (datetime.now() - start_time).total_seconds()
            logger.warning(f"⚠️ No response generated after {duration:.2f}s")
            logger.warning("="*80)
            return {
                "answer": "No response generated from agents",
                "agent_used": "None",
                "message_count": 0,
                "context": []
            }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Multi-agent query failed after {duration:.2f}s: {str(e)}")
        logger.error("="*80)
        print(f"Error in multi_agent_query_events: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing multi-agent query: {str(e)}")

@app.get("/agents")
async def list_agents():
    """List available agents in the multi-agent system"""
    try:
        global multi_agents
        if not multi_agents:
            multi_agents = create_school_events_agents()
        
        agents_info = [
            {
                "name": "WebSearch",
                "description": "Research assistant for up-to-date web information using Tavily",
                "tools": ["tavily_search_results_json"]
            },
            {
                "name": "LocalEvents",
                "description": "Local school events specialist with access to school events database",
                "tools": ["school_events_search"]
            }
        ]
        
        return {
            "agents": agents_info,
            "count": len(agents_info),
            "routing": "Automatic based on query content"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

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
        
        # Import RAGAS
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
            
            # Query the RAG using the active method
            global active_retrieval_method, original_retriever, naive_retriever, original_chain, naive_chain
            
            if active_retrieval_method == "original":
                # Original method: manually retrieve and pass to chain
                retrieved_docs = original_retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                response = original_chain.invoke({
                    "context": context,
                    "question": question
                })
            else:  # naive
                # Naive method: LCEL chain handles retrieval internally
                result = naive_chain.invoke({"question": question})
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
        
    except Exception as e:
        logger.error(f"❌ RAGAS evaluation failed: {str(e)}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error running RAGAS evaluation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")