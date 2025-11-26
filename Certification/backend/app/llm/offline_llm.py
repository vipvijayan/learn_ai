"""Helper for exposing offline Ollama LLM for all use cases."""

import logging
from app.llm.ollama_llm import OllamaLLMManager

logger = logging.getLogger(__name__)

# Global Ollama manager instance
_ollama_manager_instance = None


def get_offline_llm():
    """Get offline Ollama LLM for all tasks (counting, agents, RAG)."""
    global _ollama_manager_instance
    
    if _ollama_manager_instance is None:
        logger.info("Initializing Ollama LLM manager")
        _ollama_manager_instance = OllamaLLMManager()
        
    return _ollama_manager_instance.get_llm()


def get_offline_agent_llm():
    """Get offline LLM for agent operations (same as get_offline_llm now)."""
    return get_offline_llm()
