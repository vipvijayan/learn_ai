#!/usr/bin/env python3
"""
Gmail Authentication Script
Authenticates Gmail API access and saves token for the application.
"""

import os
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Paths
CREDENTIALS_DIR = os.path.join(os.path.dirname(__file__), 'credentials')
CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, 'gmail_credentials.json')
TOKEN_FILE = os.path.join(CREDENTIALS_DIR, 'gmail_token.json')


def authenticate_gmail():
    """Authenticate Gmail API and save token"""
    creds = None
    
    # Check if token already exists
    if os.path.exists(TOKEN_FILE):
        print("📧 Existing Gmail token found. Loading...")
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"❌ Error: Credentials file not found at: {CREDENTIALS_FILE}")
                print("\nPlease:")
                print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
                print("2. Create OAuth 2.0 credentials (Desktop app)")
                print("3. Download as 'gmail_credentials.json'")
                print(f"4. Place in: {CREDENTIALS_DIR}/")
                sys.exit(1)
            
            print("🌐 Opening browser for Gmail authentication...")
            print("Please sign in and grant permission to read your emails.")
            
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save credentials
        print(f"💾 Saving token to: {TOKEN_FILE}")
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    # Test the connection
    print("\n🧪 Testing Gmail API connection...")
    try:
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        email_address = profile.get('emailAddress')
        
        print(f"✅ Successfully authenticated as: {email_address}")
        print(f"📬 Total messages: {profile.get('messagesTotal', 0)}")
        print(f"📧 Threads total: {profile.get('threadsTotal', 0)}")
        print("\n🎉 Gmail authentication complete!")
        print(f"Token saved to: {TOKEN_FILE}")
        print("\nYou can now use Gmail in your multi-agent queries!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gmail API: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("📧 Gmail API Authentication")
    print("=" * 60)
    print()
    
    success = authenticate_gmail()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Ready to query Gmail!")
        print("=" * 60)
        print("\nExample queries:")
        print('  • "What Round Rock school events are in my email?"')
        print('  • "Did I receive any emails about soccer programs?"')
        print('  • "Show me school communications from my inbox"')
        sys.exit(0)
    else:
        sys.exit(1)
