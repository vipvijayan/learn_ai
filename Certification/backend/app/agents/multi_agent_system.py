"""
Multi-Agent Implementation for School Assistant
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

from app.tools.school_events_tool import create_school_events_tool
from app.tools.gmail_tool import create_gmail_tools
from app.tools.tavily_tool import create_school_tavily_tool, get_tavily_client

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
    Create specialized agents for the school assistant application.
    
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
    tavily_tool = create_school_tavily_tool()
    logger.info(f"   ✅ School-Context Tavily Search Tool created (max_results=5)")
    
    school_events_tool = create_school_events_tool()
    logger.info(f"   ✅ School Events Search Tool created")
    
    gmail_tools = create_gmail_tools()
    logger.info(f"   ✅ Gmail Tools created ({len(gmail_tools)} tools)")
    
    logger.info("\n🤖 Creating Agents...")
    
    # Agent 1: Search Agent (uses Tavily for web search)
    logger.info("\n--- Agent 1: WebSearch Agent ---")
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You search for K-12 school-related information on the web. "
        "Focus on: school programs, events, announcements, policies, schedules, activities, resources, and any school-related updates. "
        "Provide relevant information from official school sources when available. "
        "\n"
        "FORMAT YOUR RESPONSE:\n"
        "Line 1: [Source: Web Search]\n"
        "Line 2: Brief intro (1 sentence)\n"
        "Then list relevant information clearly and concisely.\n"
        "For events/programs:\n"
        "1. Title (Date if available)\n"
        "   • Organizer: Name\n"
        "   • Type: Category\n"
        "   • Details: Brief description\n"
        "   • Link: URL if available\n"
        "\n"
        "Keep it concise and relevant to the query."
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="WebSearch")
    logger.info("   ✅ WebSearch agent configured")
    
    # Agent 2: Local Events Agent (uses custom school events tool)
    logger.info("\n--- Agent 2: LocalEvents Agent ---")
    local_events_agent = create_agent(
        llm,
        [school_events_tool],
        "You search the local database for K-12 school-related information. "
        "Return any relevant content that matches the query including events, announcements, programs, policies, or updates. "
        "If no match, say: 'No matching information found in local database.'"
        "\n"
        "FORMAT YOUR RESPONSE:\n"
        "Line 1: [Source: Local Database]\n"
        "Line 2: Brief intro (1 sentence)\n"
        "Then present the relevant information clearly.\n"
        "For structured items (events/programs):\n"
        "1. Title\n"
        "   • Key details in bullet points\n"
        "   • Contact info if available\n"
        "\n"
        "Be concise and relevant."
    )
    local_events_node = functools.partial(agent_node, agent=local_events_agent, name="LocalEvents")
    logger.info("   ✅ LocalEvents agent configured")
    
    # Agent 3: Gmail Agent (uses Gmail MCP tools)
    logger.info("\n--- Agent 3: Gmail Agent ---")
    gmail_agent = create_agent(
        llm,
        gmail_tools,
        "You search Gmail for school-related emails. "
        "Look for any K-12 school communications including events, announcements, updates, policies, schedules, or other school information. "
        "Search using relevant keywords based on the user's query and school names."
        "\n"
        "FORMAT YOUR RESPONSE:\n"
        "Line 1: [Source: Gmail]\n"
        "Line 2: Brief intro (1 sentence)\n"
        "Then list relevant information:\n"
        "1. Subject/Topic (Date if available)\n"
        "   • From: Sender name\n"
        "   • Summary: Key information (1-2 sentences)\n"
        "   • Details: Important dates, times, locations, or action items\n"
        "\n"
        "Keep it brief and relevant to the query."
    )
    gmail_node = functools.partial(agent_node, agent=gmail_agent, name="GmailAgent")
    logger.info("   ✅ Gmail agent configured")
    
    logger.info("\n" + "="*80)
    logger.info("✅ MULTI-AGENT SYSTEM INITIALIZATION COMPLETE")
    logger.info(f"   Total Agents: 3 (WebSearch, LocalEvents, GmailAgent)")
    logger.info(f"   Total Tools: {2 + len(gmail_tools)} (Tavily, SchoolEventsSearch, Gmail)")
    logger.info("="*80 + "\n")
    
    return {
        "search_agent": search_agent,
        "search_node": search_node,
        "local_events_agent": local_events_agent,
        "local_events_node": local_events_node,
        "gmail_agent": gmail_agent,
        "gmail_node": gmail_node,
        "tools": {
            "tavily": tavily_tool,
            "school_events": school_events_tool,
            "gmail": gmail_tools
        }
    }


def create_simple_agent_graph():
    """
    Create a sequential agent graph with fallback strategy:
    1. Try Gmail first (search emails for school-related information)
    2. If no useful results, try LocalEvents (search local database)
    3. If still no useful results, fall back to WebSearch (Tavily)
    
    Returns:
        Compiled LangGraph
    """
    agents = create_school_events_agents()
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("GmailAgent", agents["gmail_node"])
    workflow.add_node("LocalEvents", agents["local_events_node"])
    workflow.add_node("WebSearch", agents["search_node"])
    
    # Router to check if Gmail found useful results
    def check_gmail_results(state):
        """
        Check if Gmail found useful information.
        If not, route to LocalEvents to search local database.
        """
        messages = state["messages"]
        last_message = messages[-1]
        content = last_message.content.lower() if last_message.content else ""
        
        logger.info(f"\n📧 CHECKING GMAIL RESULTS")
        logger.info(f"   Response length: {len(content)} chars")
        logger.info(f"   Response preview: {content[:200] if content else '(empty)'}")
        logger.info(f"   Content type: {type(last_message.content)}")
        logger.info(f"   Content value: {repr(last_message.content)}")
        
        # Check if response is empty or too short
        if not content or len(content.strip()) < 20:
            logger.info(f"   ❌ Gmail response is empty or too short ({len(content)} chars)")
            logger.info(f"   ➡️  Checking LocalEvents database")
            return "LocalEvents"
        
        # Check for indicators that no results were found OR authentication issues
        no_results_indicators = [
            "no emails found",
            "no relevant emails",
            "no emails matching",
            "gmail api error",
            "error calling gmail",
            "error:",
            "unable to",
            "couldn't find",
            "could not find",
            "no response from gmail",
            "i don't have",
            "i do not have",
            "no information",
            "not available",
            "gmail not authenticated",
            "authentication",
            "permission",
            "no results"
        ]
        
        # Check for positive indicators that results were found
        # These are very specific to actual email content
        positive_indicators = [
            "subject:",
            "from:",
            "date:",
            "email content:",
            "snippet:",
            "sender:",
            "received:",
            "message id:"
        ]
        
        has_no_results = any(indicator in content for indicator in no_results_indicators)
        has_positive_results = any(indicator in content for indicator in positive_indicators)
        
        logger.info(f"   has_no_results: {has_no_results}")
        logger.info(f"   has_positive_results: {has_positive_results}")
        
        # Gmail should only be considered successful if it has positive indicators AND no error indicators
        if has_no_results or not has_positive_results:
            logger.info(f"   ❌ Gmail search failed or found no useful results")
            if has_no_results:
                logger.info(f"   Detected error or no-results indicator")
            if not has_positive_results:
                logger.info(f"   No specific email content indicators found")
            logger.info(f"   ➡️  Checking LocalEvents database")
            return "LocalEvents"
        
        logger.info(f"   ✅ Gmail search found useful results")
        logger.info(f"   ➡️  Ending search (no fallback needed)")
        return END
    
    # Router to check LocalEvents results
    def check_local_results(state):
        """
        Check if LocalEvents found useful information.
        If not, route to WebSearch as final fallback.
        """
        messages = state["messages"]
        last_message = messages[-1]
        content = last_message.content.lower()
        
        logger.info(f"\n� CHECKING LOCAL RESULTS")
        logger.info(f"   Response preview: {content[:150]}...")
        
        # Check for indicators that no results were found
        no_results_indicators = [
            "i don't have",
            "i couldn't find",
            "no information",
            "unable to",
            "not available",
            "can't find",
            "cannot find",
            "issue with accessing",
            "unable to retrieve",
            "no specific information",
            "i'm currently unable",
            "there are no",
            "no events",
            "no programs",
            "no sports events",
            "appears that there are no",
            "it seems that there",
            "unfortunately",
            "not found",
            "no results",
            "no matching events",
            "may be related",
            "may include",
            "might be",
            "could be",
            "loosely relate"
        ]
        
        if any(indicator in content for indicator in no_results_indicators):
            logger.info(f"   ❌ Local search found no useful results")
            logger.info(f"   ➡️  Falling back to: WebSearch (Tavily)")
            return "WebSearch"
        
        logger.info(f"   ✅ Local search found useful results")
        logger.info(f"   ➡️  Ending search")
        return END
    
    # Set entry point - always start with Gmail
    logger.info("\n🔧 WORKFLOW CONFIGURATION:")
    logger.info("   Strategy: Sequential Search with Multiple Fallbacks")
    logger.info("   1️⃣  First: GmailAgent (search email inbox)")
    logger.info("   2️⃣  Second: LocalEvents (search local database)")
    logger.info("   3️⃣  Fallback: WebSearch (if both Gmail and local find nothing)")
    
    workflow.set_entry_point("GmailAgent")
    
    # After Gmail, check results and decide whether to try LocalEvents
    workflow.add_conditional_edges(
        "GmailAgent",
        check_gmail_results,
        {
            "LocalEvents": "LocalEvents",
            END: END
        }
    )
    
    # After LocalEvents, check results and decide whether to fallback to web search
    workflow.add_conditional_edges(
        "LocalEvents",
        check_local_results,
        {
            "WebSearch": "WebSearch",
            END: END
        }
    )
    
    # WebSearch always ends
    workflow.add_edge("WebSearch", END)
    
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
    
    # Log all messages for debugging
    for i, msg in enumerate(result.get('messages', [])):
        msg_type = type(msg).__name__
        msg_name = getattr(msg, 'name', 'N/A')
        msg_content_preview = msg.content[:100] if msg.content else '(empty)'
        logger.info(f"   Message {i+1}: {msg_type} | name={msg_name} | content={msg_content_preview}...")
    
    logger.info("🟢"*40 + "\n")
    
    return result


async def query_with_agent_stream(question: str, callback):
    """
    Query using the agent graph with streaming updates.
    
    Args:
        question: User's question
        callback: Async function to call with updates (agent_name, content, is_final, tool_name)
        
    Returns:
        Agent's final response
    """
    logger.info("\n" + "🔵"*40)
    logger.info(f"📝 NEW STREAMING QUERY RECEIVED: {question}")
    logger.info("🔵"*40 + "\n")
    
    graph = create_simple_agent_graph()
    
    start_time = datetime.now()
    
    # Send initial status with progress
    await callback("system", "🚀 Initiating search...", False, "initialization")
    await callback("system", "🚀 Searching: Gmail → Local Database → Web Search", False, "pipeline")
    await callback("system", "🚀 Starting intelligent search across all sources...", False, "starting")
    
    # Track which agents we've seen and their order
    agents_processing = set()
    agent_order = ["GmailAgent", "LocalEvents", "WebSearch"]
    agents_completed = []
    
    # Stream the graph execution
    try:
        result = None
        async for event in graph.astream({
            "messages": [HumanMessage(content=question)]
        }):
            logger.info(f"📡 Stream event: {list(event.keys())}")
            
            # Extract node name and messages from event
            for node_name, node_data in event.items():
                if node_name == "__start__" or node_name == "__end__":
                    continue
                
                # Map agent name to friendly name and tool
                agent_map = {
                    "GmailAgent": {"name": "Gmail", "tool": "Gmail API", "icon": "📧", "step": 1},
                    "LocalEvents": {"name": "Local Database", "tool": "Vector Database", "icon": "💾", "step": 2},
                    "WebSearch": {"name": "Web Search", "tool": "", "icon": "🌐", "step": 3}
                }
                
                agent_info = agent_map.get(node_name, {"name": node_name, "tool": "Unknown", "icon": "🔧", "step": 0})
                
                # Send agent start status if first time seeing this agent
                if node_name not in agents_processing:
                    agents_processing.add(node_name)
                    
                    # Calculate progress
                    total_agents = len(agent_order)
                    current_step = agent_info.get('step', 0)
                    progress_percent = int((current_step / total_agents) * 100)
                    
                    # Show progress bar
                    progress_bar = "█" * (current_step) + "░" * (total_agents - current_step)
                    
                    await callback(
                        "system",
                        f"[Step {current_step}/{total_agents}] {progress_bar} {progress_percent}%",
                        False,
                        f"progress_{node_name}"
                    )
                    
                    await callback(
                        "system",
                        f"{agent_info['icon']} Querying {agent_info['name']} using {agent_info['tool']}...",
                        False,
                        f"agent_start_{node_name}"
                    )
                    
                # Get messages from the node data
                if isinstance(node_data, dict) and "messages" in node_data:
                    messages = node_data["messages"]
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        content = last_message.content if last_message.content else ""
                        
                        if content:
                            # Send processing status
                            await callback(
                                "system",
                                f"✨ Processing results from {agent_info['name']}...",
                                False,
                                f"processing_{node_name}"
                            )
                            
                            # Mark agent as completed
                            if node_name not in agents_completed:
                                agents_completed.append(node_name)
                                completed_count = len(agents_completed)
                                await callback(
                                    "system",
                                    f"✅ {agent_info['name']} completed ({completed_count}/{len(agent_order)} sources searched)",
                                    False,
                                    f"completed_{node_name}"
                                )
                            
                            # Send the actual content
                            logger.info(f"   📤 Sending update from {agent_info['name']}: {len(content)} chars")
                            await callback(agent_info['name'], content, False, agent_info['tool'])
                        
                        result = node_data
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Send completion status
        await callback("system", "🎉 All sources searched successfully!", False, "all_complete")
        await callback("system", f"📊 Total time: {duration:.2f}s | Sources: {len(agents_completed)}", False, "summary")
        await callback("system", "✅ Compiling final comprehensive answer...", False, "finalizing")
        
        # Send final message
        if result and "messages" in result:
            last_message = result["messages"][-1]
            agent_name = getattr(last_message, 'name', 'Unknown')
            
            agent_map = {
                "GmailAgent": {"name": "Gmail", "tool": "Gmail API"},
                "LocalEvents": {"name": "Local Database", "tool": "Vector Database"},
                "WebSearch": {"name": "Web Search", "tool": ""}
            }
            agent_info = agent_map.get(agent_name, {"name": "Unknown", "tool": "Unknown"})
            
            await callback(agent_info['name'], last_message.content, True, agent_info['tool'], duration)
        
        logger.info("\n" + "🟢"*40)
        logger.info(f"✅ STREAMING QUERY COMPLETED (Total time: {duration:.2f}s)")
        logger.info("🟢"*40 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Streaming query failed: {str(e)}")
        await callback("error", f"Error: {str(e)}", True)
        raise



