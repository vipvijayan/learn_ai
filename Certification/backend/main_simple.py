from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from typing import List, Dict, Any
import nest_asyncio

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import tiktoken
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

app = FastAPI(title="School Events RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: List[str]

# Global variables for RAG components
vector_store = None
retriever = None
generator_chain = None

def tiktoken_len(text):
    """Calculate token length using tiktoken"""
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)
    return len(tokens)

def load_school_events_data():
    """Load and process all JSON files from the data directory"""
    documents = []
    data_dir = "../data"  # Path to your data folder
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load all JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert JSON to readable text for RAG
        content = format_event_data(data)
        doc = Document(
            page_content=content,
            metadata={"source": json_file, "type": "school_event"}
        )
        documents.append(doc)
    
    return documents

def format_event_data(data):
    """Convert JSON data to readable text format"""
    text_parts = []
    
    if "program_name" in data:
        text_parts.append(f"Program: {data['program_name']}")
    if "event_name" in data:
        text_parts.append(f"Event: {data['event_name']}")
    if "organization" in data:
        text_parts.append(f"Organization: {data['organization']}")
    
    if "event_description" in data:
        text_parts.append(f"Description: {data['event_description']}")
    
    if "target_audience" in data:
        audience = data['target_audience']
        if isinstance(audience, dict):
            if "grades" in audience:
                text_parts.append(f"Target Grades: {audience['grades']}")
            if "age_description" in audience:
                text_parts.append(f"Age Group: {audience['age_description']}")
    
    if "registration" in data:
        reg = data['registration']
        if isinstance(reg, dict):
            if "status" in reg:
                text_parts.append(f"Registration Status: {reg['status']}")
    
    # Add more structured information
    text_parts.append(f"Full Details: {json.dumps(data, indent=2)}")
    
    return "\n".join(text_parts)

def setup_rag_pipeline():
    """Initialize the RAG pipeline with embeddings and retriever"""
    global vector_store, retriever, generator_chain
    
    try:
        print("Loading school events data...")
        documents = load_school_events_data()
        print(f"Loaded {len(documents)} documents")
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100,
            length_function=tiktoken_len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not found. Using mock responses.")
            return setup_mock_rag()
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Setup chat prompt
        HUMAN_TEMPLATE = """
        #CONTEXT:
        {context}

        QUERY:
        {query}

        You are a helpful assistant for parents looking for school events and programs. 
        Use the provided context to answer the user's query about school events, programs, 
        schedules, registration, and related details. 
        
        Focus on providing clear, helpful information that helps parents make decisions 
        about their children's activities.
        
        If you don't know the answer or it's not in the provided context, 
        respond with "I don't have information about that in the current school events data."
        """
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("human", HUMAN_TEMPLATE)
        ])
        
        # Initialize chat model
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        
        # Create generator chain
        generator_chain = chat_prompt | chat_model | StrOutputParser()
        
        print("RAG pipeline initialized successfully using OpenAI!")
        
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        setup_mock_rag()

def setup_mock_rag():
    """Setup mock RAG for demo purposes when OpenAI is not available"""
    global retriever, generator_chain
    
    class MockRetriever:
        def invoke(self, question):
            # Return mock documents
            mock_docs = [
                Document(page_content="CodeWizardsHQ Logic Challenge - Free coding event for grades 3-12"),
                Document(page_content="School Holiday Day Camps - All inclusive camps with sports and STEM activities"),
                Document(page_content="Art programs available at Cordovan Art School")
            ]
            return mock_docs
    
    class MockGenerator:
        def invoke(self, inputs):
            question = inputs.get('query', '').lower()
            if 'coding' in question:
                return "I found CodeWizardsHQ Logic Challenge - a free nationwide coding event for students in grades 3-12 with weekly logic challenges and tech prizes!"
            elif 'camp' in question:
                return "School Holiday Day Camps are available with convenient hours, sports activities, STEM stations, and family discounts."
            elif 'art' in question:
                return "Cordovan Art School offers various art programs for different age groups."
            else:
                return "I can help you find information about coding programs, holiday camps, art classes, and more school events. Please ask about specific programs!"
    
    retriever = MockRetriever()
    generator_chain = MockGenerator()
    print("Mock RAG pipeline initialized for demo purposes!")

@app.get("/")
async def root():
    return {"message": "School Events RAG API is running!", "status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_events(request: QueryRequest):
    """Process user query and return RAG response"""
    try:
        # Initialize RAG pipeline if not already done
        global retriever, generator_chain
        if not retriever or not generator_chain:
            setup_rag_pipeline()
        
        if not retriever or not generator_chain:
            raise HTTPException(status_code=500, detail="RAG pipeline initialization failed")
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(request.question)
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate response
        if hasattr(generator_chain, 'invoke'):
            response = generator_chain.invoke({
                "query": request.question, 
                "context": context
            })
        else:
            # Mock response for demo
            response = f"Based on the available school events data, I found relevant information about your query: {request.question}"
        
        # Extract context snippets for display
        context_snippets = [doc.page_content[:200] + "..." for doc in retrieved_docs[:3]]
        
        return QueryResponse(
            answer=response,
            context=context_snippets
        )
        
    except Exception as e:
        print(f"Error in query_events: {e}")
        # Return a helpful error message
        return QueryResponse(
            answer=f"I encountered an error processing your request: {str(e)}. The demo mode is available for testing.",
            context=["Error occurred - using fallback response"]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "rag_initialized": vector_store is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)