"""
Multi-Agent Implementation for School Assistant
Using LangGraph and custom agents similar to Multi_Agent_RAG_LangGraph pattern
"""
import functools
import operator
import logging
import re
from typing import Annotated, List, TypedDict, Sequence, Optional
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.tools.school_events_tool import create_school_events_tool
from app.tools.gmail_tool import create_gmail_tools
from app.tools.tavily_tool import create_school_tavily_tool, get_tavily_client
from app.prompts_agents import SEARCH_AGENT_PROMPT, GMAIL_AGENT_PROMPT
from app.config import config
from app.llm.offline_llm import get_offline_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LLM_COUNT_SYSTEM_PROMPT = (
    "You analyze agent outputs that summarize search or email results. "
    "Return a single non-negative integer representing how many distinct "
    "results are described. Answer with digits only."
)
LLM_COUNT_MAX_CHARS = 5000

# State definition for agent team
class AgentState(TypedDict):
    """State for the agent team"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def _count_results_with_offline_llm(content: str) -> Optional[int]:
    """Use the locally running LLM to count distinct results if enabled."""
    if not content.strip():
        logger.info("      ✅ Offline LLM count: empty content -> 0 results")
        return 0

    try:
        offline_llm = get_offline_llm()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("      ⚠️ Offline LLM unavailable: %s", exc)
        return None

    trimmed_content = content[-LLM_COUNT_MAX_CHARS:]
    messages = [
        SystemMessage(content=LLM_COUNT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Agent response:\n" + trimmed_content + "\n\n"
                "Return the integer count of distinct results."
            )
        ),
    ]

    try:
        response = offline_llm.invoke(messages)
    except Exception as exc:  # pragma: no cover - depends on runtime
        logger.warning("      ⚠️ Offline LLM count failed: %s", exc)
        return None

    response_text = (response.content or "").strip()
    match = re.search(r"-?\d+", response_text)
    if not match:
        logger.warning(
            "      ⚠️ Offline LLM returned non-numeric response: %s",
            response_text,
        )
        return None

    result_count = int(match.group(0))
    if result_count < 0:
        result_count = 0

    logger.info("      ✅ Offline LLM detected %s results", result_count)
    return result_count


def count_results_in_content(content: str) -> int:
    """
    Count the number of results in agent response.
    Looks for patterns like "Found X results", "Result 1:", "Result 2:", etc.
    """
    if config.ENABLE_OFFLINE_LLM_FOR_RESULT_COUNTING:
        llm_count = _count_results_with_offline_llm(content)
        if llm_count is not None:
            return llm_count
        logger.info("      ⚠️ Falling back to heuristic counting (offline LLM failed)")

    logger.info(f"      🔍 Analyzing content ({len(content)} chars) for result count...")
    
    # Method 1: Look for "Found X" pattern (results, emails, items, etc.)
    found_pattern = re.search(r'found\s+(\d+)\s+(?:relevant|results?|emails?|items?|messages?|programs?)', content.lower())
    if found_pattern:
        count = int(found_pattern.group(1))
        logger.info(f"      ✅ Method 1: Detected {count} results from 'Found X' pattern")
        return count
    
    # Method 2: Count result markers like "Result 1:", "Result 2:", etc.
    result_markers = re.findall(r'(?:^|\n)(?:result|📋 result)\s+(\d+):', content.lower())
    if result_markers:
        count = len(result_markers)
        logger.info(f"      ✅ Method 2: Detected {count} results from result markers")
        return count
    
    # Method 3: Count numbered list items (1., 2., 3., etc.)
    # Look for patterns like "1. From:" which is Gmail's format
    numbered_items = re.findall(r'(?:^|\n)(\d+)\.\s+(?:From:|Subject:|[A-Z])', content)
    if numbered_items:
        count = len(numbered_items)
        logger.info(f"      ✅ Method 3: Detected {count} results from numbered list (Gmail format)")
        return count
    
    # Fallback: Count any numbered items (less strict)
    numbered_items_generic = re.findall(r'(?:^|\n)(\d+)\.\s+', content)
    if numbered_items_generic:
        count = len(numbered_items_generic)
        logger.info(f"      ✅ Method 3b: Detected {count} results from generic numbered list")
        return count
    
    # Method 4: Count bullet-point email results
    # Gmail agent formats with: • Subject/Topic (Date)
    # Count main bullet points at start of lines (emails, not nested sub-bullets)
    # Look for bullet points followed by capital letters (email entries)
    main_bullets = re.findall(r'(?:^|\n)•\s+[A-Z]', content)
    if main_bullets:
        count = len(main_bullets)
        logger.info(f"      ✅ Method 4a: Detected {count} results from bullet-point format (main bullets)")
        logger.info(f"      Sample matches: {main_bullets[:3]}")
        return count
    
    # Alternative: Look for "• Subject" or "• Topic" patterns
    subject_markers = re.findall(r'•\s*(?:subject|topic)', content.lower())
    if subject_markers:
        count = len(subject_markers)
        logger.info(f"      ✅ Method 4b: Detected {count} results from subject/topic markers")
        return count
    
    # Method 5: Check for explicit "no results" indicators
    no_results_indicators = [
        "no relevant", "no results", "no information", "couldn't find",
        "unable to", "not available", "no matching", "no emails found",
        "no events", "no programs", "could not find", "i could not find",
        "did not find", "didn't find", "no data", "no records",
        "nothing found", "found nothing", "no matches"
    ]
    for indicator in no_results_indicators:
        if indicator in content.lower():
            logger.info(f"      ⚠️ Method 5: Detected 0 results (found indicator: '{indicator}')")
            return 0
    
    # Default: assume at least 1 result if response is substantial
    if len(content.strip()) > 100:
        logger.info(f"      ⚠️ Method 6 (fallback): Assuming 1 result (substantial response, no clear pattern)")
        logger.info(f"      Content preview: {content[:200]}")
        return 1
    
    logger.info(f"      ⚠️ Method 7 (fallback): Detected 0 results (short/empty response)")
    return 0


def agent_node(state, agent, name):
    """
    Helper function to create an agent node.
    Each agent is wrapped in this function to standardize the interface.
    Each agent only sees the ORIGINAL user query, not other agents' responses.
    """
    logger.info(f"🤖 AGENT INVOKED: {name}")
    logger.info(f"   Input messages count: {len(state.get('messages', []))}")
    
    # Filter state to only include the original user query (first message)
    # This prevents agents from seeing each other's responses
    original_messages = [msg for msg in state.get('messages', []) if isinstance(msg, HumanMessage) and not hasattr(msg, 'name')]
    if original_messages:
        filtered_state = {"messages": [original_messages[0]]}  # Only the original user query
        logger.info(f"   Filtered to original query only: '{original_messages[0].content[:100]}'")
    else:
        filtered_state = state
        logger.warning(f"   Could not filter to original query, using full state")
    
    start_time = datetime.now()
    result = agent.invoke(filtered_state)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"✅ AGENT COMPLETED: {name} (took {duration:.2f}s)")
    
    if "messages" in result and len(result["messages"]) > 0:
        # Check all messages for tool outputs (more reliable for counting)
        logger.info(f"   Total messages in result: {len(result['messages'])}")
        
        # Look for tool messages (ToolMessage) that contain raw tool output
        tool_output_count = 0
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            logger.info(f"   Message {i}: {msg_type}")
            if hasattr(msg, 'content') and msg.content:
                # Check if this is a tool output with "Found X" pattern
                import re
                # Try to find "Found X emails/results/items" pattern
                found_match = re.search(r'found\s+(\d+)\s+(?:emails?|results?|items?|messages?|programs?)', msg.content.lower())
                if found_match:
                    tool_output_count = int(found_match.group(1))
                    logger.info(f"   📧 Found tool output: {tool_output_count} results from 'Found X' pattern")
                    break  # Use first match found
        
        last_message = result["messages"][-1]
        content_preview = last_message.content[:300] if last_message.content else ""
        logger.info(f"   Response preview: {content_preview}...")
        logger.info(f"   Full response length: {len(last_message.content)} chars")
        
        # Count results - prefer tool output count if available
        if tool_output_count > 0:
            result_count = tool_output_count
            logger.info(f"   ✅ Using tool output count: {result_count} results")
        else:
            result_count = count_results_in_content(last_message.content)
            logger.info(f"   ✅ Final count for {name}: {result_count} results")
        
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
    
    # Initialize LLM for agents 
    llm = None
    try:
        if config.ENABLE_OFFLINE_LLM_FOR_AGENTS:
            # Try to use offline Ollama LLM for agents (supports tool binding)
            from app.llm.offline_llm import get_offline_agent_llm
            llm = get_offline_agent_llm()
            logger.info(f"📊 Agent LLM Model: Ollama Llama-3.2-1B (offline, with tool binding support)")
        else:
            raise Exception("Offline agents disabled, using OpenAI")
            
    except Exception as e:
        # Fallback to OpenAI if offline LLM fails
        logger.warning(f"⚠️  Offline agent LLM not available ({e}), falling back to OpenAI")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        logger.info(f"📊 Agent LLM Model: gpt-4o-mini (temperature=0, fallback)")
    
    # Log if offline LLM is enabled for other parts of the system
    if config.ENABLE_OFFLINE_LLM_FOR_SEARCH:
        logger.info("🔧 Note: Offline LLM (Ollama) is also enabled for RAG pipeline")
    
    # Create tools
    logger.info("\n🔨 Creating Tools...")
    tavily_tool = create_school_tavily_tool()
    logger.info(f"   ✅ School-Context Tavily Search Tool created (max_results=5)")
    
    # school_events_tool = create_school_events_tool()
    # logger.info(f"   ✅ School Events Search Tool created")
    
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
        SEARCH_AGENT_PROMPT,
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="WebSearch")
    logger.info("   ✅ WebSearch agent configured")
    
    # LocalEvents agent and tool removed
    
    # Agent 3: Gmail Agent (uses Gmail MCP tools)
    logger.info("\n--- Agent 3: Gmail Agent ---")
    gmail_agent = create_agent(
        llm,
        gmail_tools,
        GMAIL_AGENT_PROMPT,
    )
    gmail_node = functools.partial(agent_node, agent=gmail_agent, name="GmailAgent")
    logger.info("   ✅ Gmail agent configured")
    
    logger.info("\n" + "="*80)
    logger.info("✅ MULTI-AGENT SYSTEM INITIALIZATION COMPLETE")
    logger.info(f"   Total Agents: 2 (WebSearch, GmailAgent)")
    logger.info(f"   Total Tools: {1 + len(gmail_tools)} (Tavily, Gmail)")
    logger.info("="*80 + "\n")
    
    return {
        "search_agent": search_agent,
        "search_node": search_node,
        "gmail_agent": gmail_agent,
        "gmail_node": gmail_node,
        "tools": {
            "tavily": tavily_tool,
            "gmail": gmail_tools
        }
    }


def create_simple_agent_graph(agents=None, user_email: str = None):
    """
    Create a sequential agent graph with fallback strategy:
    1. Try Gmail first (search emails for school-related information)
    2. If no useful results, fall back to WebSearch (Tavily)
    
    Args:
        agents: Pre-created agents dict (if None, will create new ones without user context)
        user_email: User's email for Gmail authentication
        
    Returns:
        Compiled LangGraph
    """
    if agents is None:
        agents = create_school_events_agents(user_email=user_email)
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("GmailAgent", agents["gmail_node"])
    workflow.add_node("WebSearch", agents["search_node"])
    
    # Router to check if Gmail found useful results
    def check_gmail_results(state):
        """
        Check if Gmail found useful information.
        If less than 5 results, route to WebSearch to search the web.
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
            logger.info(f"   ➡️  Continuing to WebSearch to find more results")
            return "WebSearch"
        
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
            logger.info(f"   ➡️  Continuing to WebSearch")
            return "WebSearch"
        
        # Check if we have at least 5 results
        if result_count < 5:
            logger.info(f"   ⚠️  Gmail found {result_count} results (less than 5)")
            logger.info(f"   ➡️  Continuing to WebSearch to find more results")
            return "WebSearch"
        
        logger.info(f"   ✅ Gmail search found {result_count} results (sufficient)")
        logger.info(f"   ➡️  Ending search (no fallback needed)")
        return END
    
    # Set entry point - always start with Gmail
    logger.info("\n🔧 WORKFLOW CONFIGURATION:")
    logger.info("   Strategy: Sequential Search with Result Count Checking")
    logger.info("   1️⃣  First: GmailAgent (search email inbox)")
    logger.info("   2️⃣  Fallback: WebSearch (if Gmail < 5 results, search web)")
    logger.info("   🎯 Goal: Find at least 5 results from Gmail, or fallback to web search")
    
    workflow.set_entry_point("GmailAgent")
    
    # After Gmail, check results and decide whether to try WebSearch
    workflow.add_conditional_edges(
        "GmailAgent",
        check_gmail_results,
        {
            "WebSearch": "WebSearch",
            END: END
        }
    )
    
    # WebSearch always ends
    workflow.add_edge("WebSearch", END)
    
    return workflow.compile()


def query_with_agent(question: str, user_email: str = None):
    """
    Query using the agent graph.
    
    Args:
        question: User's question
        user_email: User's email for Gmail authentication
        
    Returns:
        Agent's response
    """
    logger.info("\n" + "🔵"*40)
    logger.info(f"📝 NEW QUERY RECEIVED: {question}")
    logger.info(f"👤 User Email: {user_email}")
    logger.info("🔵"*40 + "\n")
    
    graph = create_simple_agent_graph(user_email=user_email)
    
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
    await callback("system", "🚀Starting search...", False, "initialization")
    
    # Track which agents we've seen and their order
    agents_processing = set()
    agent_order = ["GmailAgent", "WebSearch"]
    agents_completed = []
    # Collect all agent responses for final combination
    collected_responses = {}
    # Track total result count across all agents
    total_result_count = 0
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
                        result_count = last_message.additional_kwargs.get("result_count", 0) if hasattr(last_message, 'additional_kwargs') else 0
                        
                        # Add to total result count
                        total_result_count += result_count
                        logger.info(f"   📊 Agent {agent_info['name']} returned {result_count} results (total now: {total_result_count})")
                        
                        # Collect response for final combination (only if has results)
                        if content and result_count > 0:
                            collected_responses[node_name] = {
                                "agent": node_name,
                                "display_name": agent_info['name'],
                                "content": content,
                                "result_count": result_count,
                                "tool": agent_info['tool']
                            }
                            logger.info(f"   📦 Collected response from {agent_info['name']}: {len(content)} chars, {result_count} results")
                        else:
                            logger.info(f"   ⏭️  Skipping response from {agent_info['name']} (result_count={result_count})")
                        
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
                        result = node_data
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Send completion status
        await callback("system", "🎉 All sources searched successfully!", False, "all_complete")
        await callback("system", f"📊 Total time: {duration:.2f}s | Sources: {len(agents_completed)}", False, "summary")
        await callback("system", "✅ Compiling final comprehensive answer...", False, "finalizing")
        
        # Build final combined response from collected_responses
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 BUILDING FINAL COMBINED RESPONSE")
        logger.info(f"{'='*80}")
        logger.info(f"   Collected responses: {len(collected_responses)}")
        
        # Build final combined content from collected responses
        if len(collected_responses) > 0:
            final_content = ""
            
            for node_name, resp in collected_responses.items():
                display_name = resp['display_name']
                content = resp['content'].strip()
                result_count = resp['result_count']
                
                logger.info(f"   ✅ Including {display_name}: {result_count} results, {len(content)} chars")
                
                # Add section header
                icon = '📧' if display_name == 'Gmail' else '🌐' if display_name == 'Web Search' else '💾'
                final_content += f"### {icon} {display_name} Search Results\n\n"
                final_content += content
                final_content += "\n\n---\n\n"
            
            # Remove trailing separator
            if final_content.endswith("\n\n---\n\n"):
                final_content = final_content[:-7]
            
            logger.info(f"\n📊 FINAL RESULT COUNT:")
            logger.info(f"   📈 Total:          {total_result_count} results")
            logger.info(f"   📊 Sending ONE final combined message with total count: {total_result_count}")
            
            # Send ONE final combined message
            source_name = "Combined Results" if len(collected_responses) > 1 else list(collected_responses.values())[0]['display_name']
            tool_name = "Multiple Sources" if len(collected_responses) > 1 else list(collected_responses.values())[0]['tool']
            
            await callback(source_name, final_content, True, tool_name, duration, total_result_count, None)
        else:
            # No responses collected - send error
            logger.warning("   ⚠️ No valid responses collected from agents")
            await callback("System", "No results found from any source.", True, "System", duration, 0, None)
        
        logger.info("\n" + "🟢"*40)
        logger.info(f"✅ STREAMING QUERY COMPLETED (Total time: {duration:.2f}s)")
        logger.info("🟢"*40 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Streaming query failed: {str(e)}")
        await callback("error", f"Error: {str(e)}", True)
        raise



