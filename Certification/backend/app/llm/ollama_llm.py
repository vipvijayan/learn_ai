"""
Ollama-based offline LLM that supports tool binding for agents.
Uses Llama-3.2-1B model through Ollama for fast local inference with tool support.
"""

import logging
import subprocess
import time
from typing import Optional

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "llama3.2:1b"  # Fast 1B parameter model
OLLAMA_BASE_URL = "http://localhost:11434"

class OllamaLLMManager:
    """Manager for Ollama-based LLM with tool binding support."""
    
    def __init__(self):
        self._llm: Optional[ChatOllama] = None
        self._is_initialized = False
    
    def _ensure_ollama_running(self) -> bool:
        """Check if Ollama service is running and start if needed."""
        try:
            # Check if Ollama is running
            result = subprocess.run(
                ["curl", "-s", f"{OLLAMA_BASE_URL}/api/tags"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("✅ Ollama service is already running")
                return True
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            logger.error("❌ curl command not found - cannot check Ollama status")
            return False
        
        # Try to start Ollama
        logger.info("🚀 Starting Ollama service...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for service to start
            for i in range(10):
                time.sleep(1)
                try:
                    result = subprocess.run(
                        ["curl", "-s", f"{OLLAMA_BASE_URL}/api/tags"],
                        capture_output=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        logger.info("✅ Ollama service started successfully")
                        return True
                except subprocess.TimeoutExpired:
                    continue
            
            logger.error("❌ Ollama service failed to start within timeout")
            return False
            
        except FileNotFoundError:
            logger.error("❌ Ollama not found. Please install Ollama first: https://ollama.ai/")
            return False
        except Exception as e:
            logger.error(f"❌ Error starting Ollama service: {e}")
            return False
    
    def _ensure_model_available(self) -> bool:
        """Check if the model is available and pull if needed."""
        try:
            # Check if model exists
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and MODEL_NAME in result.stdout:
                logger.info(f"✅ Model {MODEL_NAME} is available")
                return True
            
            # Pull the model
            logger.info(f"📥 Pulling model {MODEL_NAME} (this may take a few minutes)...")
            result = subprocess.run(
                ["ollama", "pull", MODEL_NAME],
                timeout=600  # 10 minute timeout for model download
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Model {MODEL_NAME} pulled successfully")
                return True
            else:
                logger.error(f"❌ Failed to pull model {MODEL_NAME}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout while pulling model {MODEL_NAME}")
            return False
        except FileNotFoundError:
            logger.error("❌ Ollama command not found")
            return False
        except Exception as e:
            logger.error(f"❌ Error with model {MODEL_NAME}: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize the Ollama LLM."""
        if self._is_initialized:
            return True
        
        logger.info("🤖 Initializing Ollama LLM...")
        
        # Ensure Ollama service is running
        if not self._ensure_ollama_running():
            return False
        
        # Ensure model is available
        if not self._ensure_model_available():
            return False
        
        # Create the LLM instance
        try:
            self._llm = ChatOllama(
                model=MODEL_NAME,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                num_predict=256,  # Limit response length for speed
            )
            
            # Test the LLM
            logger.info("🧪 Testing Ollama LLM connection...")
            test_response = self._llm.invoke("Hello")
            logger.info(f"✅ Ollama LLM test successful: {test_response.content[:50]}...")
            
            self._is_initialized = True
            logger.info(f"🎉 Ollama LLM ({MODEL_NAME}) initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
            return False
    
    def get_llm(self) -> Optional[ChatOllama]:
        """Get the initialized LLM instance."""
        if not self._is_initialized:
            if not self.initialize():
                return None
        return self._llm


# Global instance
_ollama_manager = OllamaLLMManager()


def get_ollama_llm() -> Optional[ChatOllama]:
    """Get the Ollama LLM instance, initializing if needed."""
    return _ollama_manager.get_llm()


def is_ollama_available() -> bool:
    """Check if Ollama LLM is available and working."""
    try:
        llm = get_ollama_llm()
        return llm is not None
    except Exception:
        return False