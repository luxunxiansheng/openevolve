"""
Program Extractor Module

Handles extraction of programs from LLM responses in various formats.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExtractedProgram:
    """Result of program extraction from LLM response"""

    success: bool
    program: str = ""
    changes: Optional[List[Dict[str, str]]] = None
    error: str = ""

    def __post_init__(self):
        if self.changes is None:
            self.changes = []


class ProgramExtractor:
    """Extracts generated programs from LLM responses"""

    @staticmethod
    def extract_full_rewrite(response: str, language: str = "python") -> ExtractedProgram:
        """Extract program from full rewrite response"""
        # Try to find code blocks
        code_block_pattern = rf"```{language}\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return ExtractedProgram(success=True, program=matches[0].strip())

        # Fallback to any code block
        code_block_pattern = r"```(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return ExtractedProgram(success=True, program=matches[0].strip())

        # If no code blocks, return the response as-is
        return ExtractedProgram(success=True, program=response.strip())

    @staticmethod
    def extract_diff_changes(response: str) -> ExtractedProgram:
        """Extract changes from diff-based response"""
        # Try JSON first (preferred format)
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                changes = []
                for item in parsed:
                    if isinstance(item, dict) and "search" in item and "replace" in item:
                        changes.append(
                            {
                                "search": item["search"],
                                "replace": item["replace"],
                                "rationale": item.get("rationale", ""),
                            }
                        )
                return ExtractedProgram(success=True, changes=changes)
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback to SEARCH/REPLACE format
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(diff_pattern, response, re.DOTALL)

        if matches:
            changes = []
            for search, replace in matches:
                changes.append(
                    {
                        "search": search.strip(),
                        "replace": replace.strip(),
                        "rationale": "Extracted from SEARCH/REPLACE format",
                    }
                )
            return ExtractedProgram(success=True, changes=changes)

        return ExtractedProgram(success=False, error="No valid changes found")

    @staticmethod
    def apply_diff_changes(original_program: str, changes: List[Dict[str, str]]) -> str:
        """Apply diff changes to original program"""
        result = original_program

        for change in changes:
            search_text = change.get("search", "")
            replace_text = change.get("replace", "")

            if search_text and search_text in result:
                result = result.replace(search_text, replace_text, 1)

        return result
