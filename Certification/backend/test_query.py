#!/usr/bin/env python3
"""Quick test script for the multi-agent query endpoint"""

import requests
import json

def test_query():
    url = "http://localhost:8000/multi-agent-query"
    data = {"question": "Are there any events related to Dia de los Muertos"}
    
    print("🔍 Testing query: Are there any events related to Dia de los Muertos")
    print("="*80)
    
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✅ Response received:")
        print(f"Agent used: {result.get('agent', 'unknown')}")
        print(f"Answer: {result.get('answer', 'No answer')[:500]}...")
        print(f"\nFull response:\n{json.dumps(result, indent=2)}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to backend on http://localhost:8000")
        print("Please start the backend server first:")
        print("  cd /Users/vipinvijayan/Developer/projects/AI/AIMakerSpace/code/learn_ai_0/Certification/backend")
        print("  PATH='venv/bin:$PATH' uvicorn main:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_query()
