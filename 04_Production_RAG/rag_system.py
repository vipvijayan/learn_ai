#!/usr/bin/env python3
"""
LangChain and LangGraph RAG System
Converted from Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG.ipynb

This script demonstrates building a RAG system using LangChain, LCEL, and LangGraph
to answer questions about how people use AI.

INSTALLATION REQUIREMENTS:
==========================

1. Install Ollama:
   curl https://ollama.ai/install.sh | sh

2. Pull required models:
   ollama pull gpt-oss:20b
   ollama pull embeddinggemma:latest

3. Install Python packages:
   pip install langchain==0.3.27 langchain-community==0.3.29 langchain-core==0.3.76
   pip install langchain-ollama==0.3.8 langchain-qdrant==0.2.1 langchain-text-splitters==0.3.11
   pip install qdrant-client==1.15.1 langgraph==0.6.7 langgraph-checkpoint==2.1.1
   pip install tiktoken==0.11.0 nest-asyncio==1.6.0 typing-extensions==4.15.0
   pip install pydantic==2.11.9 numpy==2.3.3 pymupdf==1.26.4 ipython==9.5.0

   OR install from requirements file:
   pip install -r requirements_rag.txt

4. Create data directory and add PDF files:
   mkdir data
   # Add your PDF files to the data/ directory

5. Run the script:
   python rag_system.py

ENVIRONMENT SETUP (Alternative using uv):
=========================================
   uv add langchain langchain-community langchain-core langchain-ollama
   uv add langchain-qdrant qdrant-client langgraph tiktoken nest-asyncio
   uv add typing-extensions pydantic numpy pymupdf ipython
"""

import os
import getpass
import nest_asyncio
import tiktoken
from typing_extensions import TypedDict
from IPython.display import Markdown, display

# LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# LangGraph imports
from langgraph.graph import START, StateGraph

def setup_langsmith(enable=False):
    """Optional: Set up LangSmith tracing"""
    if enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
        os.environ["LANGCHAIN_PROJECT"] = "RAG-Assignment"
        
        print("LangSmith tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2", "false"))
        print("Project name:", os.getenv("LANGCHAIN_PROJECT", "Not set"))

def setup_async():
    """Enable nested async loops for Jupyter compatibility"""
    nest_asyncio.apply()

# Define State for LangGraph
class State(TypedDict):
    question: str
    context: list[Document]
    response: str

def tiktoken_len(text):
    """Token length function using tiktoken"""
    # Using cl100k_base encoding which is a good general-purpose tokenizer
    # This works well for estimating token counts even with Ollama models
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)
    return len(tokens)

def load_documents(data_directory="data"):
    """Load PDF documents from data directory"""
    directory_loader = DirectoryLoader(
        data_directory, 
        glob="**/*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    
    ai_usage_knowledge_resources = directory_loader.load()
    print(f"Loaded {len(ai_usage_knowledge_resources)} documents")
    
    # Show sample content
    if ai_usage_knowledge_resources:
        print("\nSample content (first 1000 chars):")
        print(ai_usage_knowledge_resources[0].page_content[:1000])
    
    return ai_usage_knowledge_resources

def chunk_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    
    ai_usage_knowledge_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(ai_usage_knowledge_chunks)} chunks")
    
    return ai_usage_knowledge_chunks

def setup_vector_store(chunks):
    """Set up Qdrant vector store with embeddings"""
    # Initialize embedding model
    embedding_model = OllamaEmbeddings(model="embeddinggemma:latest")
    
    # Embedding dimension for embeddinggemma
    embedding_dim = 768
    
    # Create Qdrant client (in-memory for development)
    client = QdrantClient(":memory:")
    
    # Create collection
    collection_created = client.create_collection(
        collection_name="ai_usage_knowledge_index",
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    print(f"Collection created: {collection_created}")
    
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="ai_usage_knowledge_index",
        embedding=embedding_model,
    )
    
    # Add documents to vector store
    print("Adding documents to vector store...")
    vector_store.add_documents(documents=chunks)
    print("Documents added successfully")
    
    return vector_store, embedding_model

def create_retriever(vector_store):
    """Create retriever from vector store"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Test retriever
    print("\nTesting retriever...")
    test_results = retriever.invoke("How do people use AI in their daily work?")
    print(f"Retrieved {len(test_results)} documents for test query")
    
    return retriever

def setup_chat_components():
    """Set up chat prompt template and model"""
    # Create chat prompt template
    HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context response with "I don't know"
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    
    # Test prompt
    test_prompt = chat_prompt.invoke({
        "context": "OUR CONTEXT HERE", 
        "query": "OUR QUERY HERE"
    }).messages[0].content
    print("Sample prompt format:")
    print(test_prompt[:200] + "...")
    
    # Initialize chat model
    ollama_chat_model = ChatOllama(model="gpt-oss:20b", temperature=0.6)
    
    # Test model with simple prompt
    print("\nTesting chat model...")
    test_response = ollama_chat_model.invoke(
        chat_prompt.invoke({
            "context": "Paris is the capital of France", 
            "query": "What is the capital of France?"
        })
    )
    print(f"Test response type: {type(test_response)}")
    
    # Create generator chain with output parser
    generator_chain = chat_prompt | ollama_chat_model | StrOutputParser()
    
    # Test complete chain
    chain_test = generator_chain.invoke({
        "context": "Paris is the capital of France", 
        "query": "What is the capital of France?"
    })
    print(f"Chain test result: {chain_test}")
    
    return chat_prompt, ollama_chat_model, generator_chain

def create_graph_nodes(retriever, chat_prompt, ollama_chat_model):
    """Create LangGraph nodes for RAG pipeline"""
    
    def retrieve(state: State) -> State:
        """Retrieve relevant documents"""
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state: State) -> State:
        """Generate response using LLM"""
        generator_chain = chat_prompt | ollama_chat_model | StrOutputParser()
        response = generator_chain.invoke({
            "query": state["question"], 
            "context": state["context"]
        })
        return {"response": response}
    
    return retrieve, generate

def build_graph(retrieve_node, generate_node):
    """Build and compile LangGraph"""
    # Start with the blank canvas
    graph_builder = StateGraph(State)
    
    # Add sequence of nodes
    graph_builder = graph_builder.add_sequence([retrieve_node, generate_node])
    
    # Connect START node to retrieve node
    graph_builder.add_edge(START, "retrieve")
    
    # Compile graph
    graph = graph_builder.compile()
    
    print("Graph compiled successfully!")
    print("Graph structure:", graph)
    
    return graph

def test_rag_system(graph):
    """Test the complete RAG system with sample queries"""
    
    test_queries = [
        "What are the most common ways people use AI in their work?",
        "Do people use AI for their personal lives?",
        "What concerns or challenges do people have when using AI?",
        "Who is Batman?"  # Test query outside of context
    ]
    
    print("\n" + "="*50)
    print("TESTING RAG SYSTEM")
    print("="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Question: {query}")
        print("-" * 30)
        
        response = graph.invoke({"question": query})
        print(f"Response: {response['response']}")
        print()

def main():
    """Main function to run the complete RAG pipeline"""
    print("Setting up LangChain and LangGraph RAG System")
    print("=" * 50)
    
    # Optional: Setup LangSmith (set to True if you want to enable it)
    setup_langsmith(enable=False)
    
    # Setup async for nested loops
    setup_async()
    
    # Load and process documents
    print("\n1. Loading documents...")
    documents = load_documents()
    
    print("\n2. Chunking documents...")
    chunks = chunk_documents(documents)
    
    # Setup vector store and embeddings
    print("\n3. Setting up vector store...")
    vector_store, embedding_model = setup_vector_store(chunks)
    
    # Create retriever
    print("\n4. Creating retriever...")
    retriever = create_retriever(vector_store)
    
    # Setup chat components
    print("\n5. Setting up chat components...")
    chat_prompt, ollama_chat_model, generator_chain = setup_chat_components()
    
    # Create graph nodes
    print("\n6. Creating graph nodes...")
    retrieve_node, generate_node = create_graph_nodes(
        retriever, chat_prompt, ollama_chat_model
    )
    
    # Build graph
    print("\n7. Building LangGraph...")
    graph = build_graph(retrieve_node, generate_node)
    
    # Test the system
    print("\n8. Testing RAG system...")
    test_rag_system(graph)
    
    print("\n" + "="*50)
    print("RAG System setup complete!")
    print("You can now use graph.invoke({'question': 'your question here'}) to query the system")
    print("="*50)
    
    return graph

if __name__ == "__main__":
    # Run the main function
    graph = main()
    
    # Interactive mode - uncomment to enable
    # print("\nEntering interactive mode...")
    # while True:
    #     user_question = input("\nEnter your question (or 'quit' to exit): ")
    #     if user_question.lower() == 'quit':
    #         break
    #     
    #     response = graph.invoke({"question": user_question})
    #     print(f"\nResponse: {response['response']}")