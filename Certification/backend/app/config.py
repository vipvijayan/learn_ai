import os

# Global configuration for School Assistant API

class AppConfig:
    # Set to False to disable RAGAS evaluation endpoints globally
    ENABLE_RAGAS_EVALUATION = False
    # Toggle to force offline/local LLM usage when available
    ENABLE_OFFLINE_LLM = True
    # Toggle to use offline LLM specifically for result counting heuristics
    ENABLE_OFFLINE_LLM_FOR_RESULT_COUNTING = True
    # Toggle to use offline LLM for search queries and content generation
    ENABLE_OFFLINE_LLM_FOR_SEARCH = True
    # Toggle to use offline LLM for agent operations
    ENABLE_OFFLINE_LLM_FOR_AGENTS = False

# Singleton config instance
config = AppConfig()
