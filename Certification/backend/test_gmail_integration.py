#!/usr/bin/env python3
"""
Test script for Gmail MCP integration
Tests both direct Gmail tool access and multi-agent integration
"""

import sys
import os

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

from app.tools.gmail_tool import search_gmail_emails, get_gmail_email_content
from app.agents.multi_agent_system import query_with_agent


def test_gmail_search():
    """Test Gmail search functionality"""
    print("\n" + "="*80)
    print("TEST 1: Gmail Search Tool")
    print("="*80)
    
    print("\n📧 Searching Gmail for 'Round Rock' emails...")
    try:
        # Invoke the tool directly using its func attribute
        result = search_gmail_emails.func(query="Round Rock", max_results=5)
        print("\n✅ Search Result:")
        print(result)
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_multi_agent():
    """Test multi-agent system with Gmail integration"""
    print("\n" + "="*80)
    print("TEST 2: Multi-Agent System with Gmail")
    print("="*80)
    
    test_questions = [
        "What Round Rock school events do I have in my email?",
        "Are there any soccer programs in Round Rock?",
        "What school activities are available?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 80)
        
        try:
            result = query_with_agent(question)
            
            if "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                print(f"\n✅ Answer:\n{final_message.content}")
            else:
                print("\n⚠️  No response received")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "🔵"*40)
    print("Gmail MCP Integration Test Suite")
    print("🔵"*40)
    
    # Check if credentials exist
    creds_path = os.path.join(backend_path, "credentials", "gmail_credentials.json")
    token_path = os.path.join(backend_path, "credentials", "gmail_token.json")
    
    print("\n📋 Pre-flight Checks:")
    print(f"   Credentials file: {'✅' if os.path.exists(creds_path) else '❌'} {creds_path}")
    print(f"   Token file: {'✅' if os.path.exists(token_path) else '⚠️  (will be created)'} {token_path}")
    
    if not os.path.exists(creds_path):
        print("\n❌ ERROR: Gmail credentials not found!")
        print("   Please follow the setup guide in GMAIL_MCP_SETUP.md")
        print("   Step 2: Create OAuth credentials and save to credentials/gmail_credentials.json")
        return
    
    # Test 1: Direct Gmail search
    print("\n" + "="*80)
    success = test_gmail_search()
    
    if not success:
        print("\n⚠️  Gmail search failed. Please check:")
        print("   1. Credentials are correctly configured")
        print("   2. Gmail API is enabled in Google Cloud Console")
        print("   3. You've authenticated (run: python mcp_servers/gmail_server.py)")
        return
    
    # Test 2: Multi-agent integration
    test_multi_agent()
    
    print("\n" + "🟢"*40)
    print("Test Suite Complete")
    print("🟢"*40 + "\n")


if __name__ == "__main__":
    main()
