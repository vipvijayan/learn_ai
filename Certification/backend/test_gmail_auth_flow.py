#!/usr/bin/env python
"""
Test Gmail authentication flow end-to-end
"""
import sys
sys.path.insert(0, '.')

from app.database import get_user_gmail_token
from app.agents.multi_agent_system import create_school_events_agents

print("="*80)
print("GMAIL AUTHENTICATION TEST")
print("="*80)

# Test 1: Check database
print("\n[TEST 1] Checking database for user token...")
user_email = "nair.vijayan.vipin@gmail.com"
token_data = get_user_gmail_token(user_email)

if token_data:
    print(f"✅ Token found in database for: {user_email}")
    print(f"   Gmail email: {token_data['gmail_email']}")
    print(f"   Connected at: {token_data['connected_at']}")
    print(f"   Token length: {len(token_data['token'])} chars")
else:
    print(f"❌ No token found for: {user_email}")
    sys.exit(1)

# Test 2: Create agents with user email
print("\n[TEST 2] Creating multi-agent system with user email...")
try:
    agents = create_school_events_agents(user_email=user_email)
    print(f"✅ Agents created successfully")
    print(f"   Agents: {list(agents.keys())}")
except Exception as e:
    print(f"❌ Error creating agents: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test Gmail tool directly
print("\n[TEST 3] Testing Gmail tool directly...")
from app.tools.gmail_tool import get_gmail_client, create_gmail_tools

# Create tools with token
gmail_tools = create_gmail_tools(token_data['token'])
print(f"✅ Gmail tools created: {len(gmail_tools)} tools")

# Get client
client = get_gmail_client()
print(f"✅ Gmail client retrieved")

# Test service
service = client.get_gmail_service()
if service:
    print(f"✅ Gmail service created successfully!")
    try:
        profile = service.users().getProfile(userId='me').execute()
        print(f"✅ Successfully connected to Gmail!")
        print(f"   Email: {profile.get('emailAddress')}")
        print(f"   Messages Total: {profile.get('messagesTotal')}")
    except Exception as e:
        print(f"❌ Error calling Gmail API: {e}")
else:
    print(f"❌ Gmail service is None")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
