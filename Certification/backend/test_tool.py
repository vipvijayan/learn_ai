"""
Test script for the School Events Search Tool
Run this script to test the custom tool independently
"""
import os
from dotenv import load_dotenv
from school_events_tool import create_school_events_tool

# Load environment variables
load_dotenv()

def test_school_events_tool():
    """Test the custom school events tool with various queries"""
    
    print("="*80)
    print("Testing School Events Search Tool")
    print("="*80)
    
    print("\n🔧 Initializing School Events Search Tool...")
    tool = create_school_events_tool()
    
    print(f"\n📝 Tool Name: {tool.name}")
    print(f"📝 Tool Description:\n{tool.description}")
    
    # Test queries
    test_queries = [
        "What coding programs are available?",
        "Tell me about holiday day camps",
        "When is Spring Break?",
        "Are there any art classes for kids?",
        "What music programs are there?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}: {query}")
        print(f"{'='*80}")
        
        result = tool._run(query)
        print(result)
        print()

def test_tool_as_langchain_agent():
    """Demonstrate how the tool would be used in a LangChain agent"""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    
    print("\n" + "="*80)
    print("Testing Tool with LangChain Agent")
    print("="*80)
    
    # Create the tool
    tool = create_school_events_tool()
    tools = [tool]
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that helps parents find information about school events and programs.
        Use the school_events_search tool to find relevant information.
        Always provide specific details like dates, times, registration links, and contact information when available."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries
    queries = [
        "What coding programs can my 10-year-old attend?",
        "When is spring break and what activities are available?",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        result = agent_executor.invoke({"input": query})
        print(f"\nAgent Response: {result['output']}")

if __name__ == "__main__":
    print("🚀 Starting School Events Tool Tests\n")
    
    # Test 1: Direct tool usage
    test_school_events_tool()
    
    # Test 2: Agent usage (optional - uncomment if you want to test with agent)
    # print("\n\n")
    # test_tool_as_langchain_agent()
    
    print("\n✅ All tests completed!")
