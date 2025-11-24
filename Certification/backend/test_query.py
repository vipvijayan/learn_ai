#!/usr/bin/env python3
"""Quick test script for the multi-agent query endpoint"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()  # Load environment variables

from app.agents.multi_agent_system import query_with_agent
import logging

# Set up logging to see debug output
logging.basicConfig(level=logging.INFO)

def test_direct_query():
    print("🔍 Testing direct query: show me advaith's reports")
    print("="*80)
    
    try:
        result = query_with_agent("show me advaith's reports")
        
        print(f"\n✅ Query completed:")
        print(f"Messages in result: {len(result.get('messages', []))}")
        
        for i, msg in enumerate(result.get('messages', [])):
            print(f"\n--- Message {i+1} ---")
            print(f"Name: {getattr(msg, 'name', 'N/A')}")
            print(f"Content length: {len(msg.content)}")
            print(f"Content preview: {msg.content[:300]}...")
            if hasattr(msg, 'additional_kwargs'):
                print(f"Additional kwargs: {msg.additional_kwargs}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_query()
