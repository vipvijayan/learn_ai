#!/usr/bin/env python
"""
Comprehensive Gmail Integration Diagnostic Test
Tests the complete flow from database to agent to tool invocation
"""
import sys
import os
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

print("="*80)
print("GMAIL INTEGRATION DIAGNOSTIC TEST")
print("="*80)

# Test 1: Check database for Gmail tokens
print("\n[TEST 1] Database Check")
print("-" * 40)
from app.database import get_user_gmail_token
import sqlite3

conn = sqlite3.connect('db/school_assistant.db')
cursor = conn.cursor()
cursor.execute('SELECT email, gmail_email, gmail_connected_at, LENGTH(gmail_token) FROM users')
users = cursor.fetchall()
conn.close()

if not users:
    print("❌ No users in database!")
    sys.exit(1)

print(f"✅ Found {len(users)} user(s) in database:")
for user in users:
    print(f"   - {user[0]}")
    print(f"     Gmail: {user[1] or 'Not connected'}")
    print(f"     Token: {user[3] or 0} chars")

# Pick first user with Gmail token
test_user = None
for user in users:
    if user[3] and user[3] > 0:
        test_user = user[0]
        break

if not test_user:
    print("\n❌ No user has a Gmail token!")
    sys.exit(1)

print(f"\n✅ Using test user: {test_user}")

# Test 2: Retrieve token from database
print("\n[TEST 2] Token Retrieval")
print("-" * 40)
token_data = get_user_gmail_token(test_user)
if token_data:
    print(f"✅ Token retrieved successfully")
    print(f"   Length: {len(token_data['token'])} chars")
    print(f"   Gmail: {token_data['gmail_email']}")
else:
    print("❌ Failed to retrieve token!")
    sys.exit(1)

# Test 3: Create multi-agent system with user email
print("\n[TEST 3] Multi-Agent System Creation")
print("-" * 40)
from app.agents.multi_agent_system import create_school_events_agents

try:
    print(f"Creating agents for: {test_user}")
    agents = create_school_events_agents(user_email=test_user)
    print(f"✅ Agents created successfully")
    print(f"   Keys: {list(agents.keys())}")
except Exception as e:
    print(f"❌ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check Gmail tools
print("\n[TEST 4] Gmail Tools Check")
print("-" * 40)
from app.tools.gmail_tool import _gmail_client_token, _gmail_client, get_gmail_client

print(f"Global _gmail_client_token set: {bool(_gmail_client_token)}")
print(f"Global _gmail_client exists: {bool(_gmail_client)}")

if _gmail_client_token:
    print(f"   Token length: {len(_gmail_client_token)} chars")
else:
    print("   ⚠️ WARNING: Global token is None!")

# Test 5: Get Gmail client
print("\n[TEST 5] Gmail Client Retrieval")
print("-" * 40)
try:
    client = get_gmail_client()
    print(f"✅ Client retrieved")
    print(f"   Has user_token_data: {bool(client.user_token_data)}")
    if client.user_token_data:
        print(f"   Token length: {len(client.user_token_data)} chars")
except Exception as e:
    print(f"❌ Failed to get client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Get Gmail service
print("\n[TEST 6] Gmail Service Creation")
print("-" * 40)
try:
    service = client.get_gmail_service()
    if service:
        print(f"✅ Service created successfully")
        
        # Test 7: Make actual API call
        print("\n[TEST 7] Gmail API Call")
        print("-" * 40)
        try:
            profile = service.users().getProfile(userId='me').execute()
            print(f"✅ Successfully connected to Gmail API!")
            print(f"   Email: {profile.get('emailAddress')}")
            print(f"   Messages: {profile.get('messagesTotal')}")
            print(f"   Threads: {profile.get('threadsTotal')}")
        except Exception as e:
            print(f"❌ Gmail API call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Service is None - get_gmail_service() returned None")
        print(f"   This means user_token_data was None when creating service")
except Exception as e:
    print(f"❌ Service creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test search_emails directly
print("\n[TEST 8] Direct Tool Invocation")
print("-" * 40)
try:
    result = client.search_emails("school", max_results=3)
    if "not connected" in result.lower() or "unable" in result.lower():
        print(f"❌ Search failed with error message:")
        print(f"   {result[:200]}...")
    else:
        print(f"✅ Search succeeded!")
        print(f"   Result length: {len(result)} chars")
        print(f"   Preview: {result[:150]}...")
except Exception as e:
    print(f"❌ Search invocation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSTIC TEST COMPLETE")
print("="*80)
