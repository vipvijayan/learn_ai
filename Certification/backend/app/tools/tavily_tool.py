"""
Custom Tavily Search Tool with School Context
Enhances web searches with selected school district information
"""

import os
import logging
from typing import Optional
from langchain_core.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)


class SchoolContextTavilySearch:
    """Wrapper for Tavily search that includes school context"""
    
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        self.tavily = TavilyClient(api_key=api_key)
        self.school_district = None
        self.email_suffix = None
        self.school_websites = []  # List of school website domains to restrict search
    
    def set_school_context(self, district: str = None, email_suffix: str = None, websites: list = None):
        """Set the school context for searches"""
        self.school_district = district
        self.email_suffix = email_suffix
        self.school_websites = websites if websites else []
        if district:
            logger.info(f"🏫 Tavily search context set to: {district}")
        if self.school_websites:
            logger.info(f"🌐 Tavily search will be restricted to domains: {', '.join(self.school_websites)}")
    
    def search(self, query: str) -> str:
        """
        Search the web with school context included.
        
        Args:
            query: The search query
            
        Returns:
            Search results as a formatted string
        """
        # Enhance query with school district if available
        enhanced_query = query
        if self.school_district:
            # Add school district to query for more relevant results
            enhanced_query = f"{query} {self.school_district}"
            logger.info(f"🔍 Enhanced Tavily query: {enhanced_query}")
        
        # Perform search with domain restrictions if available
        try:
            search_params = {
                "query": enhanced_query,
                "max_results": 5
            }
            
            # Add domain restrictions if school websites are provided
            if self.school_websites:
                search_params["include_domains"] = self.school_websites
                logger.info(f"🌐 Restricting search to domains: {self.school_websites}")
            
            response = self.tavily.search(**search_params)
            
            # Format results similar to TavilySearchResults output
            if response and "results" in response:
                results = []
                for result in response["results"]:
                    results.append({
                        "url": result.get("url", ""),
                        "content": result.get("content", "")
                    })
                logger.info(f"✅ Tavily search completed, found {len(results)} results")
                return results
            else:
                logger.warning("⚠️ Tavily search returned no results")
                return []
        except Exception as e:
            logger.error(f"❌ Tavily search error: {e}")
            return f"Error performing web search: {str(e)}"


# Global Tavily client
_tavily_client = None


def get_tavily_client() -> SchoolContextTavilySearch:
    """Get or create Tavily client singleton"""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = SchoolContextTavilySearch()
    return _tavily_client


@tool
def search_web_with_school_context(query: str) -> str:
    """
    Search the web for information with school district context.
    
    This tool automatically includes the selected school district in web searches
    to provide more relevant results for school-specific events and information.
    
    Useful for finding:
    - School events that aren't in the database or Gmail
    - Recent announcements and updates
    - Public information about school activities
    - Community events related to the school
    
    Args:
        query: The search query (e.g., "summer camps 2024", "basketball tryouts")
    
    Returns:
        Web search results with titles, URLs, and content snippets
    """
    client = get_tavily_client()
    return client.search(query)


def create_school_tavily_tool():
    """
    Create and return the school-context Tavily search tool.
    
    Returns:
        Tavily search tool with school context support
    """
    return search_web_with_school_context
