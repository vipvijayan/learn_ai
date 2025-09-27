"""
Multi-Agent Workflows + RAG - LangGraph

A comprehensive implementation of hierarchical multi-agent teams for RAG workflows.
This includes a research team and document writing team coordinated by a meta-supervisor.

Required Dependencies (install commands):
# pip install nest-asyncio langchain-community pymupdf tiktoken langchain-openai qdrant-client langchain-core langgraph typing-extensions pathlib uuid
"""

import os
import getpass
import nest_asyncio
import tiktoken
import functools
import operator
import uuid
from typing import Any, Callable, List, Optional, TypedDict, Union, Annotated, Dict
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# LangGraph imports
from langgraph.graph import START, StateGraph, END

# Enable nested asyncio for Jupyter compatibility
nest_asyncio.apply()

# API Key Setup
def setup_api_keys():
    """Set up required API keys"""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    if "TAVILY_API_KEY" not in os.environ:
        os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY:")

# Document Loading and Processing
def load_and_process_documents():
    """Load and process PDF documents for RAG"""
    directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    how_people_use_ai_documents = directory_loader.load()
    
    def tiktoken_len(text):
        tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
        return len(tokens)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    
    how_people_use_ai_chunks = text_splitter.split_documents(how_people_use_ai_documents)
    print(f"Created {len(how_people_use_ai_chunks)} document chunks")
    return how_people_use_ai_chunks

# Create vector store and retriever
def create_vector_store(chunks):
    """Create Qdrant vector store from document chunks"""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:"
    )
    
    qdrant_retriever = qdrant_vectorstore.as_retriever()
    return qdrant_retriever, embedding_model

# RAG Chain Setup
def create_rag_chain(retriever):
    """Create a simple RAG chain"""
    HUMAN_TEMPLATE = """
    #CONTEXT:
    {context}

    QUERY:
    {query}

    Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
    """
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    
    generator_llm = ChatOpenAI(model="gpt-4o-mini")
    
    class State(TypedDict):
        question: str
        context: List[Document]
        response: str
    
    def retrieve(state: State):
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state: State):
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response = generator_chain.invoke({"query": state["question"], "context": state["context"]})
        return {"response": response}
    
    rag_graph = StateGraph(State).add_sequence([retrieve, generate])
    rag_graph.add_edge(START, "retrieve")
    compiled_rag_graph = rag_graph.compile()
    
    return compiled_rag_graph

# Helper Functions for Agent Graphs
def agent_node(state, agent, name):
    """Helper function to wrap agents as nodes"""
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason!")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(options), team_members=", ".join(members))
    
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

# Research Team Setup
def create_research_team(compiled_rag_graph):
    """Create research team with search and RAG capabilities"""
    # Tools
    tavily_tool = TavilySearchResults(max_results=5)
    
    @tool
    def retrieve_information(
        query: Annotated[str, "query to ask the retrieve information tool"]
    ):
        """Use Retrieval Augmented Generation to retrieve information about how people use AI"""
        return compiled_rag_graph.invoke({"question": query})
    
    # Research Team State
    class ResearchTeamState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        team_members: List[str]
        next: str
    
    research_llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Create agents
    search_agent = create_agent(
        research_llm,
        [tavily_tool],
        "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    
    research_agent = create_agent(
        research_llm,
        [retrieve_information],
        "You are a research assistant who can provide specific information on how people use AI",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="HowPeopleUseAIRetriever")
    
    # Supervisor
    research_supervisor_agent = create_team_supervisor(
        research_llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers:  Search, HowPeopleUseAIRetriever. Given the following user request,"
        " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. "
        " You should never ask your team to do anything beyond research. They are not required to write content or posts."
        " You should only pass tasks to workers that are specifically research focused."
        " When finished, respond with FINISH."),
        ["Search", "HowPeopleUseAIRetriever"],
    )
    
    # Build graph
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("HowPeopleUseAIRetriever", research_node)
    research_graph.add_node("ResearchSupervisor", research_supervisor_agent)
    
    research_graph.add_edge("Search", "ResearchSupervisor") 
    research_graph.add_edge("HowPeopleUseAIRetriever", "ResearchSupervisor")
    research_graph.add_conditional_edges(
        "ResearchSupervisor",
        lambda x: x["next"],
        {"Search": "Search", "HowPeopleUseAIRetriever": "HowPeopleUseAIRetriever", "FINISH": END},
    )
    research_graph.set_entry_point("ResearchSupervisor")
    
    compiled_research_graph = research_graph.compile()
    
    # Create chain interface
    def enter_research_chain(message: str):
        results = {
            "messages": [HumanMessage(content=message)],
        }
        return results
    
    research_chain = enter_research_chain | compiled_research_graph
    return research_chain

# Document Writing Team Setup
def create_document_writing_team(embedding_model):
    """Create document writing team with planning, writing, and editing capabilities"""
    
    # Load previous cohort data
    previous_cohort_loader = CSVLoader("data/AIE7_Projects_with_Domains.csv", 
                                     content_columns=["Project Domain", "Secondary Domain (if any)"])
    previous_cohort = previous_cohort_loader.load()
    
    qdrant_previous_cohort_vectorstore = Qdrant.from_documents(
        documents=previous_cohort,
        embedding=embedding_model,
        location=":memory:"
    )
    qdrant_previous_cohort_retriever = qdrant_previous_cohort_vectorstore.as_retriever()
    
    # Working directory setup
    def create_random_subdirectory():
        random_id = str(uuid.uuid4())[:8]
        subdirectory_path = os.path.join('./content/data', random_id)
        os.makedirs(subdirectory_path, exist_ok=True)
        return subdirectory_path
    
    os.makedirs('./content/data', exist_ok=True)
    WORKING_DIRECTORY = Path(create_random_subdirectory())
    
    # Document tools
    @tool
    def create_outline(
        points: Annotated[List[str], "List of main points or sections."],
        file_name: Annotated[str, "File path to save the outline."],
    ) -> Annotated[str, "Path of the saved outline file."]:
        """Create and save an outline."""
        with (WORKING_DIRECTORY / file_name).open("w") as file:
            for i, point in enumerate(points):
                file.write(f"{i + 1}. {point}\n")
        return f"Outline saved to {file_name}"
    
    @tool
    def read_document(
        file_name: Annotated[str, "File path to save the document."],
        start: Annotated[Optional[int], "The start line. Default is 0"] = None,
        end: Annotated[Optional[int], "The end line. Default is None"] = None,
    ) -> str:
        """Read the specified document."""
        with (WORKING_DIRECTORY / file_name).open("r") as file:
            lines = file.readlines()
        if start is not None:
            start = 0
        return "\n".join(lines[start:end])
    
    @tool
    def write_document(
        content: Annotated[str, "Text content to be written into the document."],
        file_name: Annotated[str, "File path to save the document."],
    ) -> Annotated[str, "Path of the saved document file."]:
        """Create and save a text document."""
        with (WORKING_DIRECTORY / file_name).open("w") as file:
            file.write(content)
        return f"Document saved to {file_name}"
    
    @tool 
    def reference_previous_responses(
        query: Annotated[str, "The query to search for in the previous responses."],
    ) -> Annotated[str, "The previous responses that match the query."]:
        """Search for previous responses that match the query."""
        return qdrant_previous_cohort_retriever.invoke(query)
    
    @tool
    def edit_document(
        file_name: Annotated[str, "Path of the document to be edited."],
        inserts: Annotated[
            Dict[int, str],
            "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
        ] = {},
    ) -> Annotated[str, "Path of the edited document file."]:
        """Edit a document by inserting text at specific line numbers."""
        with (WORKING_DIRECTORY / file_name).open("r") as file:
            lines = file.readlines()
        
        sorted_inserts = sorted(inserts.items())
        
        for line_number, text in sorted_inserts:
            if 1 <= line_number <= len(lines) + 1:
                lines.insert(line_number - 1, text + "\n")
            else:
                return f"Error: Line number {line_number} is out of range."
        
        with (WORKING_DIRECTORY / file_name).open("w") as file:
            file.writelines(lines)
        
        return f"Document edited and saved to {file_name}"
    
    # Document Writing State
    class DocWritingState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        team_members: str
        next: str
        current_files: str
    
    # Prelude function
    def prelude(state):
        written_files = []
        if not WORKING_DIRECTORY.exists():
            WORKING_DIRECTORY.mkdir()
        try:
            written_files = [
                f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
            ]
        except:
            pass
        if not written_files:
            return {**state, "current_files": "No files written."}
        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n"
            + "\n".join([f" - {f}" for f in written_files]),
        }
    
    authoring_llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Create agents
    doc_writer_agent = create_agent(
        authoring_llm,
        [write_document, edit_document, read_document],
        ("You are an expert writing customer assistance responses.\n"
        "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    doc_writing_node = functools.partial(
        agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
    )
    
    note_taking_agent = create_agent(
        authoring_llm,
        [create_outline, read_document, reference_previous_responses],
        ("You are an expert senior researcher tasked with writing a customer assistance outline and"
        " taking notes to craft a customer assistance response.\n{current_files}"),
    )
    context_aware_note_taking_agent = prelude | note_taking_agent
    note_taking_node = functools.partial(
        agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
    )
    
    copy_editor_agent = create_agent(
        authoring_llm,
        [write_document, edit_document, read_document],
        ("You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues\n"
        "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_copy_editor_agent = prelude | copy_editor_agent
    copy_editing_node = functools.partial(
        agent_node, agent=context_aware_copy_editor_agent, name="CopyEditor"
    )
    
    authoring_supervisor_agent = create_team_supervisor(
        authoring_llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers: {team_members}. You should always verify the technical"
        " contents after any edits are made. "
        "Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When each team is finished,"
        " you must respond with FINISH."),
        ["DocWriter", "NoteTaker", "CopyEditor"],
    )
    
    # Build graph
    authoring_graph = StateGraph(DocWritingState)
    authoring_graph.add_node("DocWriter", doc_writing_node)
    authoring_graph.add_node("NoteTaker", note_taking_node)
    authoring_graph.add_node("CopyEditor", copy_editing_node)
    authoring_graph.add_node("AuthoringSupervisor", authoring_supervisor_agent)
    
    authoring_graph.add_edge("DocWriter", "AuthoringSupervisor")
    authoring_graph.add_edge("NoteTaker", "AuthoringSupervisor")
    authoring_graph.add_edge("CopyEditor", "AuthoringSupervisor")
    
    authoring_graph.add_conditional_edges(
        "AuthoringSupervisor",
        lambda x: x["next"],
        {
            "DocWriter": "DocWriter",
            "NoteTaker": "NoteTaker",
            "CopyEditor": "CopyEditor",
            "FINISH": END,
        },
    )
    
    authoring_graph.set_entry_point("AuthoringSupervisor")
    compiled_authoring_graph = authoring_graph.compile()
    
    # Create chain interface
    def enter_authoring_chain(message: str, members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results
    
    authoring_chain = (
        functools.partial(enter_authoring_chain, members=authoring_graph.nodes)
        | compiled_authoring_graph
    )
    
    return authoring_chain, WORKING_DIRECTORY

# Meta-Supervisor and Full Graph
def create_meta_supervisor(research_chain, authoring_chain):
    """Create the meta-supervisor that coordinates between research and authoring teams"""
    
    super_llm = ChatOpenAI(model="gpt-4o-mini")
    
    super_supervisor_agent = create_team_supervisor(
        super_llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following teams: {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When all workers are finished,"
        " you must respond with FINISH.",
        ["Research team", "Response team"],
    )
    
    # State and helper functions
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str
    
    def get_last_message(state: State) -> str:
        return state["messages"][-1].content
    
    def join_graph(response: dict):
        return {"messages": [response["messages"][-1]]}
    
    # Build super graph
    super_graph = StateGraph(State)
    
    super_graph.add_node("Research team", get_last_message | research_chain | join_graph)
    super_graph.add_node("Response team", get_last_message | authoring_chain | join_graph)
    super_graph.add_node("SuperSupervisor", super_supervisor_agent)
    
    super_graph.add_edge("Research team", "SuperSupervisor")
    super_graph.add_edge("Response team", "SuperSupervisor")
    super_graph.add_conditional_edges(
        "SuperSupervisor",
        lambda x: x["next"],
        {
            "Response team": "Response team",
            "Research team": "Research team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("SuperSupervisor")
    
    compiled_super_graph = super_graph.compile()
    return compiled_super_graph

# Main execution function
def main():
    """Main function to run the multi-agent RAG system"""
    print("Setting up Multi-Agent RAG LangGraph system...")
    
    # Setup API keys
    setup_api_keys()
    
    # Load and process documents
    print("Loading and processing documents...")
    chunks = load_and_process_documents()
    
    # Create vector store
    print("Creating vector store...")
    retriever, embedding_model = create_vector_store(chunks)
    
    # Create RAG chain
    print("Creating RAG chain...")
    compiled_rag_graph = create_rag_chain(retriever)
    
    # Test RAG chain
    print("\nTesting RAG chain...")
    result = compiled_rag_graph.invoke({"question": "How does the average person use AI?"})
    print(f"RAG Test Result: {result['response'][:200]}...")
    
    # Create research team
    print("\nCreating research team...")
    research_chain = create_research_team(compiled_rag_graph)
    
    # Create document writing team  
    print("Creating document writing team...")
    authoring_chain, working_directory = create_document_writing_team(embedding_model)
    
    # Create meta-supervisor
    print("Creating meta-supervisor...")
    compiled_super_graph = create_meta_supervisor(research_chain, authoring_chain)
    
    # Run the full system
    print("\n" + "="*50)
    print("RUNNING MULTI-AGENT SYSTEM")
    print("="*50)
    
    query = "Write a report on the rise of context engineering in the LLM Space in 2025, and how it's impacting how people are using AI."
    
    for s in compiled_super_graph.stream(
        {
            "messages": [HumanMessage(content=query)],
        },
        {"recursion_limit": 30},
    ):
        if "__end__" not in s:
            print(s)
            print("---")
    
    print(f"\nFiles created in: {working_directory}")
    print("Multi-Agent RAG system execution completed!")

if __name__ == "__main__":
    main()