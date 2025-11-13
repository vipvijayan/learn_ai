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


def count_results_in_content(content: str) -> int:
    """
    Count the number of results in agent response.
    Looks for patterns like "Found X results", "Result 1:", "Result 2:", etc.
    """
    import re
    
    # Method 1: Look for "Found X results" pattern
    found_pattern = re.search(r'found\s+(\d+)\s+(?:relevant|results?)', content.lower())
    if found_pattern:
        count = int(found_pattern.group(1))
        logger.info(f"      📊 Detected {count} results from 'Found X' pattern")
        return count
    
    # Method 2: Count result markers like "Result 1:", "Result 2:", etc.
    result_markers = re.findall(r'(?:^|\n)(?:result|📋 result)\s+(\d+):', content.lower())
    if result_markers:
        count = len(result_markers)
        logger.info(f"      📊 Detected {count} results from result markers")
        return count
    
    # Method 3: Count numbered list items (1., 2., 3., etc.)
    numbered_items = re.findall(r'(?:^|\n)(\d+)\.\s+', content)
    if numbered_items:
        count = len(numbered_items)
        logger.info(f"      📊 Detected {count} results from numbered list")
        return count
    
    # Method 4: Check for explicit "no results" indicators
    no_results_indicators = [
        "no relevant", "no results", "no information", "couldn't find",
        "unable to", "not available", "no matching", "no emails found",
        "no events", "no programs"
    ]
    if any(indicator in content.lower() for indicator in no_results_indicators):
        logger.info(f"      📊 Detected 0 results (no-results indicator)")
        return 0
    
    # Default: assume at least 1 result if response is substantial
    if len(content.strip()) > 100:
        logger.info(f"      📊 Assuming 1+ results (substantial response)")
        return 1
    
    logger.info(f"      📊 Detected 0 results (short/empty response)")
    return 0


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
        
        # Count results in the response
        result_count = count_results_in_content(last_message.content)
        
        # Store result count in message metadata for routing decisions
        message_with_metadata = HumanMessage(
            content=last_message.content,
            name=name,
            additional_kwargs={"result_count": result_count}
        )
        
        return {"messages": [message_with_metadata]}
    
    logger.warning(f"⚠️  AGENT WARNING: {name} returned no messages")
    return {"messages": [HumanMessage(content="No response generated", name=name, additional_kwargs={"result_count": 0})]}


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


def create_school_events_agents(user_email: str = None):
    """
    Create specialized agents for the school assistant application.
    
    Args:
        user_email: Email of the user (for per-user Gmail authentication)
    
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
    
    # Get user's Gmail token if available
    user_gmail_token = None
    if user_email:
        from app.database import get_user_gmail_token
        token_data = get_user_gmail_token(user_email)
        if token_data:
            user_gmail_token = token_data['token']
            logger.info(f"   📧 Using per-user Gmail authentication for: {user_email}")
        else:
            logger.warning(f"   ⚠️ No Gmail token found for user: {user_email}")
    else:
        logger.warning(f"   ⚠️ No user_email provided - Gmail will not be available")
    
    gmail_tools = create_gmail_tools(user_gmail_token)
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


def create_simple_agent_graph(agents=None):
    """
    Create a sequential agent graph with fallback strategy:
    1. Try Gmail first (search emails for school-related information)
    2. If no useful results, try LocalEvents (search local database)
    3. If still no useful results, fall back to WebSearch (Tavily)
    
    Args:
        agents: Pre-created agents dict (if None, will create new ones without user context)
    
    Returns:
        Compiled LangGraph
    """
    if agents is None:
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
        If less than 5 results, route to LocalEvents to search local database.
        """
        messages = state["messages"]
        last_message = messages[-1]
        content = last_message.content.lower() if last_message.content else ""
        
        logger.info(f"\n📧 CHECKING GMAIL RESULTS")
        logger.info(f"   Response length: {len(content)} chars")
        logger.info(f"   Response preview: {content[:200] if content else '(empty)'}")
        
        # Get result count from metadata
        result_count = last_message.additional_kwargs.get("result_count", 0) if hasattr(last_message, 'additional_kwargs') else 0
        logger.info(f"   📊 Result count: {result_count}")
        
        # Check if response is empty or too short
        if not content or len(content.strip()) < 20:
            logger.info(f"   ❌ Gmail response is empty or too short ({len(content)} chars)")
            logger.info(f"   ➡️  Continuing to LocalEvents to find more results")
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
            logger.info(f"   ➡️  Continuing to LocalEvents")
            return "LocalEvents"
        
        # Check if we have at least 5 results
        if result_count < 5:
            logger.info(f"   ⚠️  Gmail found {result_count} results (less than 5)")
            logger.info(f"   ➡️  Continuing to LocalEvents to find more results")
            return "LocalEvents"
        
        logger.info(f"   ✅ Gmail search found {result_count} results (sufficient)")
        logger.info(f"   ➡️  Ending search (no fallback needed)")
        return END
    
    # Router to check LocalEvents results
    def check_local_results(state):
        """
        Check if LocalEvents found useful information.
        If less than 5 total results (combining Gmail + LocalEvents), route to WebSearch.
        """
        messages = state["messages"]
        last_message = messages[-1]
        content = last_message.content.lower()
        original_query = messages[0].content.lower()
        
        logger.info(f"\n🗄️ CHECKING LOCAL RESULTS")
        logger.info(f"   Response preview: {content[:150]}...")
        
        # Get result count from current agent
        local_result_count = last_message.additional_kwargs.get("result_count", 0) if hasattr(last_message, 'additional_kwargs') else 0
        logger.info(f"   📊 LocalEvents result count: {local_result_count}")
        
        # Count total results from all previous agents
        total_result_count = local_result_count
        for msg in messages[:-1]:  # Exclude the last message (current)
            if hasattr(msg, 'additional_kwargs') and 'result_count' in msg.additional_kwargs:
                agent_count = msg.additional_kwargs['result_count']
                total_result_count += agent_count
                logger.info(f"   📊 Adding {agent_count} results from previous agent: {getattr(msg, 'name', 'Unknown')}")
        
        logger.info(f"   📊 Total results so far: {total_result_count}")
        
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
        
        # Check if query has specific year constraints
        import re
        year_pattern = r'\b(19|20)\d{2}\b'
        query_years = re.findall(year_pattern, original_query)
        
        if query_years:
            # Query has specific year(s)
            # Need to check if the ACTUAL DATA (not LLM response) contains those years
            # Look for tool call results in the messages
            tool_results_found = False
            tool_content = ""
            
            for msg in messages:
                # Check if this is a tool message
                if hasattr(msg, 'type') and msg.type == 'tool':
                    tool_results_found = True
                    tool_content += str(msg.content).lower() + " "
                # Also check additional_kwargs for tool_calls
                if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    tool_results_found = True
            
            # If we have tool results, check if they contain the requested years
            if tool_results_found and tool_content:
                data_years = re.findall(year_pattern, tool_content)
                years_in_data = any(year in data_years for year in query_years)
                
                if not years_in_data:
                    logger.info(f"   ⚠️  Query asks about specific year(s): {query_years}")
                    logger.info(f"   ⚠️  Tool results don't contain those years")
                    logger.info(f"   ❌ Local database doesn't have data for requested year")
                    logger.info(f"   ➡️  Falling back to: WebSearch (Tavily)")
                    return "WebSearch"
            else:
                # Fallback: check if response mentions years (LLM might be hallucinating)
                response_years = re.findall(year_pattern, content)
                
                # If LLM mentions the query years but we didn't find tool results,
                # it's likely hallucinating - fall back to web search
                if any(year in content for year in query_years):
                    logger.info(f"   ⚠️  Query asks about specific year(s): {query_years}")
                    logger.info(f"   ⚠️  Response mentions years but no tool results verified")
                    logger.info(f"   ⚠️  LLM may be hallucinating dates from query")
                    logger.info(f"   ❌ Cannot verify data for requested year")
                    logger.info(f"   ➡️  Falling back to: WebSearch (Tavily)")
                    return "WebSearch"
        
        if any(indicator in content for indicator in no_results_indicators):
            logger.info(f"   ❌ Local search found no useful results")
            logger.info(f"   ➡️  Falling back to: WebSearch (Tavily)")
            return "WebSearch"
        
        # Check if we have at least 5 total results
        if total_result_count < 5:
            logger.info(f"   ⚠️  Total results: {total_result_count} (less than 5)")
            logger.info(f"   ➡️  Continuing to WebSearch to find more results")
            return "WebSearch"
        
        logger.info(f"   ✅ Local search found useful results (total: {total_result_count})")
        logger.info(f"   ➡️  Ending search")
        return END
    
    # Set entry point - always start with Gmail
    logger.info("\n🔧 WORKFLOW CONFIGURATION:")
    logger.info("   Strategy: Sequential Search with Result Count Checking")
    logger.info("   1️⃣  First: GmailAgent (search email inbox)")
    logger.info("   2️⃣  Second: LocalEvents (if Gmail < 5 results, search local database)")
    logger.info("   3️⃣  Fallback: WebSearch (if total < 5 results, search web)")
    logger.info("   🎯 Goal: Accumulate at least 5 results across all sources")
    
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


async def query_with_agent_stream(question: str, callback, agents=None):
    """
    Query using the agent graph with streaming updates.
    
    Args:
        question: User's question
        callback: Async function to call with updates (agent_name, content, is_final, tool_name, duration, result_count, tool_counts)
        agents: Pre-created agents dict with user-specific credentials (optional)
        
    Returns:
        Agent's final response
    """
    logger.info("\n" + "🔵"*40)
    logger.info(f"📝 NEW STREAMING QUERY RECEIVED: {question}")
    logger.info("🔵"*40 + "\n")
    
    graph = create_simple_agent_graph(agents=agents)
    
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
                            
                            # Send the actual content with result count
                            result_count = last_message.additional_kwargs.get("result_count", 0) if hasattr(last_message, 'additional_kwargs') else 0
                            logger.info(f"   📤 Sending update from {agent_info['name']}: {len(content)} chars, {result_count} results")
                            await callback(agent_info['name'], content, False, agent_info['tool'], None, result_count)
                        
                        result = node_data
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Send completion status
        await callback("system", "🎉 All sources searched successfully!", False, "all_complete")
        await callback("system", f"📊 Total time: {duration:.2f}s | Sources: {len(agents_completed)}", False, "summary")
        await callback("system", "✅ Compiling final comprehensive answer...", False, "finalizing")
        
        # Send final message - combine all agent responses
        if result and "messages" in result:
            combined_responses = []
            total_results = 0
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 ANALYZING FINAL RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"   Total messages in state: {len(result['messages'])}")
            
            # Agent name mapping for display
            agent_display_names = {
                "GmailAgent": "Gmail",
                "LocalEvents": "Local Database",
                "WebSearch": "Web Search"
            }
            
            # First pass: count results from each agent
            # Initialize all agents with 0 counts first
            agent_result_counts = {
                "Gmail": 0,
                "Local Database": 0,
                "Web Search": 0
            }
            
            # Then update with actual counts from agents that executed
            for msg in result["messages"][1:]:
                if hasattr(msg, 'name') and msg.name in ["GmailAgent", "LocalEvents", "WebSearch"]:
                    result_count = msg.additional_kwargs.get("result_count", 0) if hasattr(msg, 'additional_kwargs') else 0
                    # Use display name for the counts
                    display_name = agent_display_names.get(msg.name, msg.name)
                    agent_result_counts[display_name] = result_count
            
            logger.info(f"\n📊 RESULT COUNT BY AGENT:")
            logger.info(f"   📧 Gmail:          {agent_result_counts.get('Gmail', 0)} results")
            logger.info(f"   💾 Local Database: {agent_result_counts.get('Local Database', 0)} results")
            logger.info(f"   🌐 Web Search:     {agent_result_counts.get('Web Search', 0)} results")
            logger.info(f"   📈 Total:          {sum(agent_result_counts.values())} results")
            logger.info(f"")
            
            # Collect all agent responses (skip the original user query)
            for i, msg in enumerate(result["messages"][1:], 1):
                msg_name = getattr(msg, 'name', None)
                msg_type = type(msg).__name__
                msg_content_preview = msg.content[:100] if msg.content else '(empty)'
                logger.info(f"   Message {i}: type={msg_type}, name={msg_name}, preview={msg_content_preview}")
                
                if hasattr(msg, 'name') and msg.name in ["GmailAgent", "LocalEvents", "WebSearch"]:
                    agent_name = msg.name
                    content = msg.content
                    
                    # Get result count
                    result_count = msg.additional_kwargs.get("result_count", 0) if hasattr(msg, 'additional_kwargs') else 0
                    
                    # Only include responses with actual results (non-empty and result_count > 0)
                    if content and len(content.strip()) > 50 and result_count > 0:
                        combined_responses.append({
                            "agent": agent_name,
                            "content": content,
                            "result_count": result_count
                        })
                        total_results += result_count
                        logger.info(f"   ✅ Including response from {agent_name} ({result_count} results, {len(content)} chars)")
                    else:
                        logger.info(f"   ⏭️  Skipping response from {agent_name} (result_count={result_count}, length={len(content) if content else 0})")
            
            # If multiple agents contributed, combine their responses
            if len(combined_responses) > 1:
                logger.info(f"   🔗 Combining {len(combined_responses)} agent responses (total: {total_results} results)")
                
                final_content = f"📊 Combined results from {len(combined_responses)} sources ({total_results} total results):\n\n"
                
                for i, resp in enumerate(combined_responses, 1):
                    source_name = agent_display_names.get(resp["agent"], resp["agent"])
                    final_content += f"{'='*70}\n"
                    final_content += f"Source {i}: {source_name} ({resp['result_count']} results)\n"
                    final_content += f"{'='*70}\n"
                    final_content += resp["content"]
                    if i < len(combined_responses):
                        final_content += "\n\n"
                
                # Build counts dict with display name - include ALL agents even with 0 results
                display_counts = agent_result_counts.copy()  # Use the complete counts we already built
                logger.info(f"   📊 Sending final callback with counts: {display_counts}")
                
                await callback("Combined Results", final_content, True, "Multiple Sources", duration, None, display_counts)
            else:
                # Single agent response
                last_message = result["messages"][-1]
                agent_name = getattr(last_message, 'name', 'Unknown')
                
                agent_map = {
                    "GmailAgent": {"name": "Gmail", "tool": "Gmail API"},
                    "LocalEvents": {"name": "Local Database", "tool": "Vector Database"},
                    "WebSearch": {"name": "Web Search", "tool": ""}
                }
                agent_info = agent_map.get(agent_name, {"name": "Unknown", "tool": "Unknown"})
                
                # Get result count for this agent
                result_count = last_message.additional_kwargs.get("result_count", 0) if hasattr(last_message, 'additional_kwargs') else 0
                
                # Build counts dict with display name
                display_counts = {agent_info['name']: result_count}
                logger.info(f"   📊 Sending final callback with counts: {display_counts}")
                
                await callback(agent_info['name'], last_message.content, True, agent_info['tool'], duration, result_count, display_counts)
        
        logger.info("\n" + "🟢"*40)
        logger.info(f"✅ STREAMING QUERY COMPLETED (Total time: {duration:.2f}s)")
        logger.info("🟢"*40 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Streaming query failed: {str(e)}")
        await callback("error", f"Error: {str(e)}", True)
        raise



