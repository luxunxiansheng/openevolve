"""
Code generation utilities for program evolution

This module handles LLM-based code generation, including response parsing
and code extraction from various response formats.
"""

import asyncio
from typing import Optional, Tuple
from openevolve.llm.llm_interface import LLMInterface


class CodeGenerator:
    """
    Handles LLM-based code generation for program evolution

    This class manages the interaction with LLMs to generate improved code,
    including response parsing and validation.
    """

    def __init__(self, llm: LLMInterface, language: str = "python", max_code_length: int = 20480):
        """
        Initialize code generator

        Args:
            llm: LLM interface for code generation
            language: Programming language for code context
            max_code_length: Maximum allowed code length
        """
        self.llm = llm
        self.language = language
        self.max_code_length = max_code_length

    async def generate_code_async(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate code asynchronously using LLM

        Args:
            system_prompt: System prompt with general instructions
            user_prompt: User prompt with specific task details

        Returns:
            Generated code string

        Raises:
            ValueError: If generated code exceeds maximum length
        """
        # Call LLM with both system and user prompts
        response = await self.llm.generate(user_prompt, system_message=system_prompt)

        # Handle ensemble response (if LLM returns multiple responses)
        if isinstance(response, list):
            response = response[0] if response else ""

        # Extract code from response
        code = self._extract_code_from_response(str(response))

        # Validate code length
        if code and len(code) > self.max_code_length:
            raise ValueError(
                f"Generated code exceeds maximum length ({len(code)} > {self.max_code_length})"
            )

        return code

    def generate_code(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate code synchronously using LLM

        Args:
            system_prompt: System prompt with general instructions
            user_prompt: User prompt with specific task details

        Returns:
            Generated code string
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self.generate_code_async(system_prompt, user_prompt))
        finally:
            loop.close()

    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from LLM response, handling various formats

        Args:
            response: Raw LLM response

        Returns:
            Extracted code string
        """
        # Look for code blocks first (most common format)
        code = self._extract_from_code_blocks(response)
        if code:
            return code

        # Look for SEARCH/REPLACE patterns (diff mode)
        if self._is_diff_response(response):
            return response.strip()

        # Fallback: return the whole response
        return response.strip()

    def _extract_from_code_blocks(self, response: str) -> Optional[str]:
        """Extract code from markdown code blocks"""
        if "```" not in response:
            return None

        start = response.find("```")
        end = response.find("```", start + 3)

        if end == -1:
            return None

        code = response[start + 3 : end]

        # Remove language identifier from first line
        lines = code.strip().split("\n")
        if lines and lines[0].strip() in ["python", "py", self.language]:
            return "\n".join(lines[1:])

        return code.strip()

    def _is_diff_response(self, response: str) -> bool:
        """Check if response contains SEARCH/REPLACE diff patterns"""
        return "<<<<<<< SEARCH" in response and ">>>>>>> REPLACE" in response

    def extract_diffs(self, response: str) -> list[Tuple[str, str]]:
        """
        Extract SEARCH/REPLACE pairs from diff response

        Args:
            response: Response containing diff patterns

        Returns:
            List of (search, replace) tuples
        """
        diffs = []
        lines = response.split("\n")

        i = 0
        while i < len(lines):
            if lines[i].strip() == "<<<<<<< SEARCH":
                # Find the separator
                separator_idx = None
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == "=======":
                        separator_idx = j
                        break

                if separator_idx is None:
                    i += 1
                    continue

                # Find the end
                end_idx = None
                for j in range(separator_idx + 1, len(lines)):
                    if lines[j].strip() == ">>>>>>> REPLACE":
                        end_idx = j
                        break

                if end_idx is None:
                    i += 1
                    continue

                # Extract search and replace blocks
                search_block = "\n".join(lines[i + 1 : separator_idx])
                replace_block = "\n".join(lines[separator_idx + 1 : end_idx])

                diffs.append((search_block, replace_block))
                i = end_idx + 1
            else:
                i += 1

        return diffs
