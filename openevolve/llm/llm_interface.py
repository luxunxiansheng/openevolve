"""
Base LLM interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional




class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        pass
