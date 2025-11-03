"""
Gmail MCP Server
A Model Context Protocol server for Gmail integration.
Provides tools to search and retrieve emails from Gmail.
"""

import os
import json
import logging
from typing import Any, Optional, List
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Paths for credentials
BASE_DIR = Path(__file__).parent.parent
CREDENTIALS_DIR = BASE_DIR / "credentials"
TOKEN_FILE = CREDENTIALS_DIR / "gmail_token.json"
CREDENTIALS_FILE = CREDENTIALS_DIR / "gmail_credentials.json"


class GmailMCPServer:
    """Gmail MCP Server implementation"""
    
    def __init__(self):
        self.server = Server("gmail-server")
        self.gmail_service = None
        
        # Register handlers
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)
        
    def get_gmail_service(self):
        """Authenticate and return Gmail service"""
        if self.gmail_service:
            return self.gmail_service
            
        creds = None
        
        # Load existing token
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"Gmail credentials file not found at {CREDENTIALS_FILE}. "
                        "Please download OAuth credentials from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            CREDENTIALS_DIR.mkdir(exist_ok=True)
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        self.gmail_service = build('gmail', 'v1', credentials=creds)
        return self.gmail_service
    
    async def list_tools(self) -> List[Tool]:
        """List available Gmail tools"""
        return [
            Tool(
                name="search_gmail",
                description=(
                    "Search Gmail messages using Gmail search operators. "
                    "Returns message subjects, snippets, senders, and dates. "
                    "Useful for finding emails about specific topics like school events, "
                    "Round Rock schools, or any other subject."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Gmail search query using Gmail search operators. "
                                "Examples: 'Round Rock', 'from:school@roundrockisd.org', "
                                "'subject:event after:2024/01/01', 'is:unread important'"
                            )
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of emails to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_email_content",
                description=(
                    "Get the full content of a specific email by message ID. "
                    "Returns the complete email body, subject, sender, and date."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The Gmail message ID to retrieve"
                        }
                    },
                    "required": ["message_id"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: dict) -> List[TextContent]:
        """Execute a tool call"""
        try:
            if name == "search_gmail":
                return await self.search_gmail(arguments)
            elif name == "get_email_content":
                return await self.get_email_content(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def search_gmail(self, arguments: dict) -> List[TextContent]:
        """Search Gmail messages"""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        
        try:
            service = self.get_gmail_service()
            
            # Search for messages
            results = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return [TextContent(
                    type="text",
                    text=f"No emails found for query: {query}"
                )]
            
            # Get details for each message
            email_summaries = []
            for msg in messages:
                msg_id = msg['id']
                message = service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                
                # Extract headers
                headers = message.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
                
                snippet = message.get('snippet', '')
                
                email_summaries.append({
                    'id': msg_id,
                    'subject': subject,
                    'from': sender,
                    'date': date,
                    'snippet': snippet
                })
            
            # Format response
            response = f"Found {len(email_summaries)} email(s) for query: {query}\n\n"
            for idx, email in enumerate(email_summaries, 1):
                response += f"[{idx}] ID: {email['id']}\n"
                response += f"    Subject: {email['subject']}\n"
                response += f"    From: {email['from']}\n"
                response += f"    Date: {email['date']}\n"
                response += f"    Preview: {email['snippet'][:200]}...\n\n"
            
            return [TextContent(type="text", text=response)]
            
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return [TextContent(
                type="text",
                text=f"Gmail API error: {str(error)}"
            )]
    
    async def get_email_content(self, arguments: dict) -> List[TextContent]:
        """Get full email content"""
        message_id = arguments.get("message_id")
        
        if not message_id:
            return [TextContent(
                type="text",
                text="Error: message_id is required"
            )]
        
        try:
            service = self.get_gmail_service()
            
            # Get full message
            message = service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = message.get('payload', {}).get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            # Extract body
            body = self._extract_body(message.get('payload', {}))
            
            response = f"Email ID: {message_id}\n"
            response += f"Subject: {subject}\n"
            response += f"From: {sender}\n"
            response += f"Date: {date}\n"
            response += f"\n--- Email Body ---\n{body}\n"
            
            return [TextContent(type="text", text=response)]
            
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return [TextContent(
                type="text",
                text=f"Gmail API error: {str(error)}"
            )]
    
    def _extract_body(self, payload: dict) -> str:
        """Extract email body from payload"""
        import base64
        
        if 'body' in payload and 'data' in payload['body']:
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    if 'data' in part.get('body', {}):
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif 'parts' in part:
                    body = self._extract_body(part)
                    if body:
                        return body
        
        return "Could not extract email body"


async def main():
    """Run the Gmail MCP server"""
    server_instance = GmailMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
