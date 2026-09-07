"""Answer generation using LLM."""
from typing import Optional
from app.config import settings
from app.rag.prompt import get_system_prompt
import logging

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str, system_prompt: str = "", 
                 max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using LLM."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq API LLM provider."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize Groq provider."""
        try:
            from groq import Groq
            
            self.api_key = api_key or settings.llm_api_key
            self.model = model or settings.llm_model
            
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not set in environment")
            
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq provider initialized with model: {self.model}")
            
        except ImportError:
            logger.error("Groq library not installed. Install with: pip install groq")
            raise
        except Exception as e:
            logger.error(f"Error initializing Groq provider: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: str = "", 
                 max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using Groq API."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            raise


class AnswerGenerator:
    """Generates answers using retrieved context and LLM."""
    
    def __init__(self, llm_provider: LLMProvider = None):
        """Initialize answer generator."""
        if llm_provider is None:
            # Initialize default provider
            if settings.llm_provider == "groq":
                llm_provider = GroqProvider()
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
        
        self.llm = llm_provider
    
    def generate_answer(self, query: str, context: str, mode: str = "lecture") -> str:
        """
        Generate an answer using query and context.
        
        Args:
            query: User's question
            context: Retrieved lecture context
            mode: "lecture" or "tutor"
        
        Returns:
            Generated answer with source citations
        """
        try:
            system_prompt = get_system_prompt(mode)
            
            user_prompt = f"""### RETRIEVED LECTURE SOURCES:

{context}

---

### USER QUESTION:
{query}

### ASSISTANT RESPONSE (include SOURCES at the end):
"""
            
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


def get_llm_provider(provider_name: str = None) -> LLMProvider:
    """Get LLM provider based on configuration."""
    provider_name = provider_name or settings.llm_provider
    
    if provider_name == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
