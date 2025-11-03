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
    
    def __init__(self):
        # Path to credentials
        self.credentials_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",  # up to app/
            "..",  # up to backend/
            "credentials"
        )
        self.token_path = os.path.join(self.credentials_dir, "gmail_token.json")
        self._service = None
    
    def get_gmail_service(self):
        """Get authenticated Gmail service"""
        if self._service:
            return self._service
            
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            
            # Check if token exists
            if not os.path.exists(self.token_path):
                logger.warning("Gmail token not found. Please authenticate first.")
                return None
            
            # Load credentials
            creds = Credentials.from_authorized_user_file(self.token_path, ['https://www.googleapis.com/auth/gmail.readonly'])
            
            # Refresh if expired
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build service
            self._service = build('gmail', 'v1', credentials=creds)
            return self._service
            
        except Exception as e:
            logger.error(f"Error getting Gmail service: {e}")
            return None
    
    def search_emails(self, query: str, max_results: int = 10) -> str:
        """Search Gmail for emails matching query"""
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
            
            # Search for messages
            results = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return f"No emails found matching: {query}"
            
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


def get_gmail_client() -> GmailToolClient:
    """Get or create Gmail client singleton"""
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailToolClient()
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


def create_gmail_tools() -> List:
    """
    Create and return Gmail tools for LangChain agents.
    
    Returns:
        List of Gmail-related tools
    """
    return [search_gmail_emails, get_gmail_email_content]
