"""
OpenAI API interface for LLMs
"""

import asyncio
from email import message
import logging
from typing import Any, Dict, List, Optional

import openai

from openevolve.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        system_message: Optional[str] = "You are a helpful assistant.",
        name: str = "Qwen3-14B-AWQ",
        temperature: float = 0.95,
        top_p: float = 1.0,
        max_tokens: int = 20480,
        timeout: int = 120,
        retries: int = 3,
        retry_delay: int = 5,
        api_base: str = "http://localhost:8010/v1",
        api_key: Optional[str] = "none",
        random_seed: Optional[int] = 0,
    ):
        self.model = name
        self.system_message = system_message
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.api_base = api_base
        self.api_key = api_key
        self.random_seed = random_seed

        # Set up API client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    async def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate text using a system message and conversational context"""
        if system_message is None:
            system_message = self.system_message

        messages = [{"role": "user", "content": prompt}]
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        if self.api_base == "https://api.openai.com/v1" and str(self.model).lower().startswith("o"):
            # For o-series models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content
