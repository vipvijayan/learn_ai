"""
Multi-Agent Implementation for School Events RAG
Using LangGraph and custom agents similar to Multi_Agent_RAG_LangGraph pattern
"""
import functools
import operator
import logging
from typing import Annotated, List, TypedDict, Sequence
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from school_events_tool import create_school_events_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# State definition for agent team
class AgentState(TypedDict):
    """State for the agent team"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def agent_node(state, agent, name):
    """
    Helper function to create an agent node.
    Each agent is wrapped in this function to standardize the interface.
    """
    logger.info(f"🤖 AGENT INVOKED: {name}")
    logger.info(f"   Input messages count: {len(state.get('messages', []))}")
    
    start_time = datetime.now()
    result = agent.invoke(state)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"✅ AGENT COMPLETED: {name} (took {duration:.2f}s)")
    
    if "messages" in result and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        logger.info(f"   Response preview: {last_message.content[:150]}...")
        return {"messages": [HumanMessage(content=last_message.content, name=name)]}
    
    logger.warning(f"⚠️  AGENT WARNING: {name} returned no messages")
    return {"messages": [HumanMessage(content="No response generated", name=name)]}


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
):
    """
    Create a React agent with tools using LangGraph's create_react_agent.
    
    Args:
        llm: The language model to power the agent
        tools: List of tools the agent can use
        system_prompt: The system prompt defining the agent's role
    
    Returns:
        Compiled agent graph
    """
    logger.info(f"🔧 Creating agent with {len(tools)} tools: {[t.name for t in tools]}")
    
    system_message = (
        f"{system_prompt}\n\n"
        "Work autonomously according to your specialty, using the tools available to you. "
        "Do not ask for clarification. "
        "Your other team members will collaborate with you with their own specialties. "
        "You are chosen for a reason! Use your tools effectively."
    )
    
    # Create react agent using LangGraph - prompt parameter sets the system message
    agent = create_react_agent(llm, tools, prompt=system_message)
    logger.info("✅ Agent created successfully")
    return agent


def create_school_events_agents():
    """
    Create specialized agents for the school events application.
    
    Returns:
        Dictionary containing all agent nodes
    """
    logger.info("="*80)
    logger.info("🚀 INITIALIZING MULTI-AGENT SYSTEM")
    logger.info("="*80)
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    logger.info(f"📊 LLM Model: gpt-4o-mini (temperature=0)")
    
    # Create tools
    logger.info("\n🔨 Creating Tools...")
    tavily_tool = TavilySearchResults(max_results=3)
    logger.info(f"   ✅ Tavily Search Tool created (max_results=3)")
    
    school_events_tool = create_school_events_tool()
    logger.info(f"   ✅ School Events Search Tool created")
    
    logger.info("\n🤖 Creating Agents...")
    
    # Agent 1: Search Agent (uses Tavily for web search)
    logger.info("\n--- Agent 1: WebSearch Agent ---")
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You are a research assistant who can search for up-to-date information using the Tavily search engine. "
        "Use this to find current information about schools, programs, organizations, or general educational topics. "
        "When searching, be specific and return relevant, factual information."
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="WebSearch")
    logger.info("   ✅ WebSearch agent configured")
    
    # Agent 2: Local Events Agent (uses custom school events tool)
    logger.info("\n--- Agent 2: LocalEvents Agent ---")
    local_events_agent = create_agent(
        llm,
        [school_events_tool],
        "You are a local school events specialist. Use the school_events_search tool to find information "
        "about local school events, programs, camps, classes, and activities from our database. "
        "Provide detailed information including dates, times, registration links, and age requirements."
    )
    local_events_node = functools.partial(agent_node, agent=local_events_agent, name="LocalEvents")
    logger.info("   ✅ LocalEvents agent configured")
    
    logger.info("\n" + "="*80)
    logger.info("✅ MULTI-AGENT SYSTEM INITIALIZATION COMPLETE")
    logger.info(f"   Total Agents: 2 (WebSearch, LocalEvents)")
    logger.info(f"   Total Tools: 2 (Tavily, SchoolEventsSearch)")
    logger.info("="*80 + "\n")
    
    return {
        "search_agent": search_agent,
        "search_node": search_node,
        "local_events_agent": local_events_agent,
        "local_events_node": local_events_node,
        "tools": {
            "tavily": tavily_tool,
            "school_events": school_events_tool
        }
    }


def create_simple_agent_graph():
    """
    Create a simple agent graph that routes between web search and local events search.
    
    Returns:
        Compiled LangGraph
    """
    agents = create_school_events_agents()
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("WebSearch", agents["search_node"])
    workflow.add_node("LocalEvents", agents["local_events_node"])
    
    # Define routing logic
    def router(state):
        """Route to appropriate agent based on the last message"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Simple routing: if message mentions "local", "school events", "programs"
        # use local events, otherwise use web search
        content = last_message.content.lower()
        
        logger.info(f"\n🔀 ROUTING DECISION")
        logger.info(f"   Query: {last_message.content[:100]}...")
        
        if any(keyword in content for keyword in [
            "local", "school event", "program", "camp", "class",
            "registration", "when is", "available", "coding program",
            "art class", "music program"
        ]):
            logger.info(f"   ➡️  Routing to: LocalEvents Agent")
            logger.info(f"   Reason: Query contains local event keywords")
            return "LocalEvents"
        else:
            logger.info(f"   ➡️  Routing to: WebSearch Agent")
            logger.info(f"   Reason: Query appears to be general/web search")
            return "WebSearch"
    
    # Set entry point
    workflow.set_conditional_entry_point(
        router,
        {
            "WebSearch": "WebSearch",
            "LocalEvents": "LocalEvents"
        }
    )
    
    # Both nodes end after execution
    workflow.add_edge("WebSearch", END)
    workflow.add_edge("LocalEvents", END)
    
    return workflow.compile()


def query_with_agent(question: str):
    """
    Query using the agent graph.
    
    Args:
        question: User's question
        
    Returns:
        Agent's response
    """
    logger.info("\n" + "🔵"*40)
    logger.info(f"📝 NEW QUERY RECEIVED: {question}")
    logger.info("🔵"*40 + "\n")
    
    graph = create_simple_agent_graph()
    
    start_time = datetime.now()
    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "🟢"*40)
    logger.info(f"✅ QUERY COMPLETED (Total time: {duration:.2f}s)")
    logger.info(f"   Messages in response: {len(result.get('messages', []))}")
    logger.info("🟢"*40 + "\n")
    
    return result


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Initializing School Events Multi-Agent System\n")
    
    # Create agents
    agents = create_school_events_agents()
    
    print("✅ Agents created:")
    print(f"  - Search Agent (with Tavily)")
    print(f"  - Local Events Agent (with school_events_search)")
    print()
    
    # Test queries
    test_queries = [
        "What coding programs are available?",  # Should use LocalEvents
        "What are the best coding bootcamps nationally?",  # Should use WebSearch
    ]
    
    print("Testing agent graph:")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        
        result = query_with_agent(query)
        
        # Print the result
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, HumanMessage):
                    print(f"💬 {msg.name}: {msg.content[:200]}...")
        print()
