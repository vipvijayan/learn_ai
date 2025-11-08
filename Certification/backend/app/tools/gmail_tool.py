"""
Gmail Tool for LangChain Integration
Direct Gmail API access without MCP subprocess calls.
"""

import os
import logging
import pickle
from typing import Optional, List, Dict, Any
from datetime import datetime
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class GmailToolClient:
    """Client for interacting with Gmail API directly"""
    
    def __init__(self, user_token_data: Optional[Dict[str, Any]] = None):
        """
        Initialize Gmail client.
        
        Args:
            user_token_data: User-specific token data from database (required for per-user auth)
        """
        # Get credentials directory (one level up from this file)
        self.credentials_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "credentials"
        )
        
        # Path for OAuth credentials file
        self.credentials_path = os.path.join(self.credentials_dir, "gmail_credentials.json")
        self._service = None
        self.email_suffixes = []  # List of email suffixes for multi-school support
        self.user_token_data = user_token_data  # Per-user token data from database
    
    def set_email_suffix(self, email_suffix: str):
        """Set a single email suffix for filtering Gmail searches (legacy support)"""
        if email_suffix:
            self.email_suffixes = [email_suffix]
    
    def set_email_suffixes(self, email_suffixes: List[str]):
        """Set multiple email suffixes for filtering Gmail searches across multiple schools"""
        self.email_suffixes = email_suffixes if email_suffixes else []
    
    def get_gmail_service(self):
        """Get authenticated Gmail service using per-user credentials only"""
        if self._service:
            return self._service
            
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            import json
            from datetime import datetime
            
            # Require user-specific token (no fallback)
            if not self.user_token_data:
                logger.warning("⚠️ No Gmail token found for user. User must sign in with Gmail.")
                return None
            
            logger.info("✅ Using per-user Gmail authentication")
            
            # Parse token data from database
            token_json = json.loads(self.user_token_data)
            
            # Create credentials from token data
            creds = Credentials(
                token=token_json.get('token'),
                refresh_token=token_json.get('refresh_token'),
                token_uri=token_json.get('token_uri'),
                client_id=token_json.get('client_id'),
                client_secret=token_json.get('client_secret'),
                scopes=token_json.get('scopes'),
            )
            
            # Set expiry if available
            if token_json.get('expiry'):
                creds.expiry = datetime.fromisoformat(token_json['expiry'])
            
            # Refresh if expired
            if creds.expired and creds.refresh_token:
                logger.info("🔄 Refreshing expired user Gmail token")
                creds.refresh(Request())
                # Note: Token refresh should update database, but for now just use refreshed creds
            
            # Build service
            self._service = build('gmail', 'v1', credentials=creds)
            return self._service
            
        except Exception as e:
            logger.error(f"Error getting Gmail service: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def search_emails(self, query: str, max_results: int = 10) -> str:
        """Search Gmail for emails matching query across all selected schools"""
        try:
            service = self.get_gmail_service()
            if not service:
                return (
                    "📧 Gmail is not connected for this user.\n\n"
                    "To search your Gmail, your account needs to be connected with Gmail access. "
                    "Since you signed in with Gmail, your account should already have access. "
                    "If you're seeing this message, please sign out and sign in again to reconnect Gmail."
                )
            
            # Build search query with multiple email suffix filters if available
            search_query = query
            if self.email_suffixes and len(self.email_suffixes) > 0:
                # Build OR query for multiple school domains
                # Gmail syntax: from:*@domain1.com OR from:*@domain2.com
                from_filters = [f"from:*@{suffix}" for suffix in self.email_suffixes]
                from_clause = f"({' OR '.join(from_filters)})"
                search_query = f"{query} {from_clause}"
                logger.info(f"Gmail search with multi-school filter: {search_query}")
                logger.info(f"Searching across {len(self.email_suffixes)} school domain(s): {', '.join(self.email_suffixes)}")
            
            # Search for messages
            results = service.users().messages().list(
                userId='me',
                q=search_query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                suffix_msg = ""
                if self.email_suffixes and len(self.email_suffixes) > 0:
                    suffix_msg = f" from schools: {', '.join(['@' + s for s in self.email_suffixes])}"
                return f"No emails found matching: {query}{suffix_msg}"
            
            # Format results with body previews
            email_list = []
            for msg in messages:
                msg_data = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'  # Changed to 'full' to get body content
                ).execute()
                
                headers = {h['name']: h['value'] for h in msg_data['payload']['headers']}
                
                # Extract body preview (first 200 chars)
                body = self._extract_body(msg_data['payload'])
                body_preview = body[:200].replace('\n', ' ').strip() if body else "No preview available"
                if len(body) > 200:
                    body_preview += "..."
                
                email_list.append({
                    'id': msg['id'],
                    'from': headers.get('From', 'Unknown'),
                    'subject': headers.get('Subject', 'No subject'),
                    'date': headers.get('Date', 'Unknown date'),
                    'preview': body_preview
                })
            
            # Format output
            output = f"Found {len(email_list)} emails matching '{query}':\n\n"
            for i, email in enumerate(email_list, 1):
                output += f"{i}. From: {email['from']}\n"
                output += f"   Subject: {email['subject']}\n"
                output += f"   Date: {email['date']}\n"
                output += f"   Preview: {email['preview']}\n"
                output += f"   ID: {email['id']}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error searching Gmail: {e}")
            return f"Error searching Gmail: {str(e)}"
    
    def get_email_content(self, message_id: str) -> str:
        """Get full content of an email by ID"""
        try:
            service = self.get_gmail_service()
            if not service:
                return (
                    "Gmail not authenticated. To enable Gmail search:\n"
                    "1. Run: python authenticate_gmail.py\n"
                    "2. Sign in with your Gmail account\n"
                    "3. Grant permission to read emails\n"
                    "See GMAIL_AUTHENTICATION.md for detailed instructions."
                )
            
            msg = service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            # Extract email details
            headers = {h['name']: h['value'] for h in msg['payload']['headers']}
            
            # Extract body
            body = self._extract_body(msg['payload'])
            
            output = f"Email ID: {message_id}\n"
            output += f"From: {headers.get('From', 'Unknown')}\n"
            output += f"Subject: {headers.get('Subject', 'No subject')}\n"
            output += f"Date: {headers.get('Date', 'Unknown date')}\n"
            output += f"\nContent:\n{body[:1000]}{'...' if len(body) > 1000 else ''}"
            
            return output
            
        except Exception as e:
            logger.error(f"Error getting email content: {e}")
            return f"Error getting email content: {str(e)}"
    
    def _extract_body(self, payload):
        """Extract email body from payload"""
        import base64
        
        try:
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/plain':
                        body_data = part.get('body', {})
                        if body_data and 'data' in body_data:
                            return base64.urlsafe_b64decode(body_data['data']).decode('utf-8')
                    elif 'parts' in part:
                        body = self._extract_body(part)
                        if body and body != "No text content found":
                            return body
            elif 'body' in payload:
                body_data = payload.get('body', {})
                if body_data and 'data' in body_data:
                    return base64.urlsafe_b64decode(body_data['data']).decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting email body: {e}")
            return f"Error extracting body: {str(e)}"
        
        return "No text content found"


# Global Gmail client
_gmail_client = None
_gmail_client_token = None  # Current user-specific token from database
_gmail_client_last_token = None  # Last token used to create client


def get_gmail_client() -> GmailToolClient:
    """Get or create Gmail client - recreates when token changes"""
    global _gmail_client, _gmail_client_token, _gmail_client_last_token
    
    # Recreate client if:
    # 1. Client doesn't exist yet
    # 2. Token has changed (different user or token updated)
    if _gmail_client is None or _gmail_client_token != _gmail_client_last_token:
        logger.info(f"🔄 Creating Gmail client for user")
        _gmail_client = GmailToolClient(user_token_data=_gmail_client_token)
        _gmail_client_last_token = _gmail_client_token
    
    return _gmail_client


@tool
def search_gmail_emails(query: str, max_results: int = 10) -> str:
    """
    Search Gmail for emails matching the query.
    
    This tool searches your Gmail inbox for emails containing specific keywords,
    from specific senders, or matching other Gmail search criteria.
    
    Useful for finding:
    - School event information from emails
    - Communications from specific schools (e.g., Round Rock ISD)
    - Event announcements, schedules, and updates
    - Any other email-based information
    
    Args:
        query: Gmail search query. Examples:
            - "Round Rock" - search for Round Rock mentions
            - "from:school@roundrockisd.org" - from specific sender
            - "subject:event after:2024/01/01" - subject and date filters
            - "Round Rock school events" - multiple keywords
        max_results: Maximum number of emails to return (default: 10)
    
    Returns:
        A formatted list of matching emails with subjects, senders, dates, and previews.
    """
    client = get_gmail_client()
    return client.search_emails(query, max_results)


@tool
def get_gmail_email_content(message_id: str) -> str:
    """
    Get the full content of a specific Gmail message.
    
    Use this tool after finding relevant emails with search_gmail_emails
    to retrieve the complete email body and details.
    
    Args:
        message_id: The Gmail message ID (obtained from search_gmail_emails)
    
    Returns:
        The complete email including subject, sender, date, and full body content.
    """
    client = get_gmail_client()
    return client.get_email_content(message_id)


def create_gmail_tools(user_token_data: Optional[str] = None) -> List:
    """
    Create and return Gmail tools for LangChain agents.
    
    Args:
        user_token_data: User-specific Gmail token JSON string from database
        
    Returns:
        List of Gmail-related tools
    """
    # Store user token in global variable (will be used by tool functions)
    global _gmail_client_token
    _gmail_client_token = user_token_data
    
    return [search_gmail_emails, get_gmail_email_content]
