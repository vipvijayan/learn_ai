#!/usr/bin/env python3
"""
Check OAuth Configuration
Verify the OAuth credentials are properly configured.
"""

import json
import os

CREDENTIALS_DIR = os.path.join(os.path.dirname(__file__), 'credentials')
CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, 'gmail_credentials.json')

print("=" * 60)
print("🔍 OAuth Configuration Check")
print("=" * 60)
print()

# Check if credentials file exists
if not os.path.exists(CREDENTIALS_FILE):
    print(f"❌ Credentials file not found: {CREDENTIALS_FILE}")
    exit(1)

print(f"✅ Credentials file found: {CREDENTIALS_FILE}")
print()

# Load and inspect credentials
with open(CREDENTIALS_FILE, 'r') as f:
    creds = json.load(f)

# Check structure
if 'installed' in creds:
    app_type = 'installed'
    config = creds['installed']
    print("✅ Application Type: Desktop App (correct)")
elif 'web' in creds:
    app_type = 'web'
    config = creds['web']
    print("⚠️  Application Type: Web App (should be Desktop App)")
else:
    print("❌ Unknown application type")
    exit(1)

print()
print("📋 OAuth Configuration Details:")
print(f"   Client ID: {config.get('client_id', 'N/A')[:50]}...")
print(f"   Project ID: {config.get('project_id', 'N/A')}")
print(f"   Auth URI: {config.get('auth_uri', 'N/A')}")
print(f"   Token URI: {config.get('token_uri', 'N/A')}")
print()

# Provide troubleshooting steps
print("=" * 60)
print("🔧 Troubleshooting Steps if Authentication Still Fails:")
print("=" * 60)
print()
print("1. **Wait 2-3 minutes** after adding test user")
print("   Google's systems need time to update")
print()
print("2. **Try in Incognito/Private browsing mode**")
print("   Clears cached authentication state")
print()
print("3. **Verify OAuth Consent Screen settings:**")
print("   • Go to: https://console.cloud.google.com/apis/credentials/consent")
print("   • Publishing status: 'Testing'")
print("   • Your email in 'Test users' section")
print("   • Scopes include: '.../auth/gmail.readonly'")
print()
print("4. **Check OAuth credentials:**")
print("   • Go to: https://console.cloud.google.com/apis/credentials")
print("   • Application type must be: 'Desktop app'")
print("   • Not 'Web application' or 'Android/iOS'")
print()
print("5. **If still not working, create NEW OAuth credentials:**")
print("   • Delete current credentials")
print("   • Create new 'OAuth 2.0 Client ID'")
print("   • Select 'Desktop app'")
print("   • Download JSON")
print(f"   • Replace: {CREDENTIALS_FILE}")
print()
print("6. **Alternative: Use 'Continue' button**")
print("   If you see a warning screen with 'Continue' option,")
print("   click it to proceed anyway (safe for personal use)")
print()
