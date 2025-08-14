"""
Code evaluation utilities for program evolution

This module handles evaluation of generated code using both execution
and LLM-based evaluators.
"""

import asyncio
from typing import Dict, Optional, Any
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator


class CodeEvaluator:
    """
    Handles evaluation of generated code using multiple evaluators

    This class coordinates execution and LLM-based evaluation of code,
    providing comprehensive metrics for program evolution.
    """

    def __init__(
        self,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        language: str = "python",
    ):
        """
        Initialize code evaluator

        Args:
            exe_evaluator: Execution-based evaluator
            llm_evaluator: Optional LLM-based evaluator
            language: Programming language for evaluation context
        """
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language

    async def evaluate_code_async(self, code: str) -> Dict[str, float]:
        """
        Evaluate code asynchronously using all available evaluators

        Args:
            code: Code to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Start with execution evaluation
        result = await self.exe_evaluator.evaluate(code=code, language=self.language)

        # Add LLM evaluation if available
        if self.llm_evaluator:
            try:
                llm_result = await self.llm_evaluator.evaluate(code=code, language=self.language)
                # Add LLM metrics with prefix to avoid conflicts
                for key, value in llm_result.items():
                    result[f"llm_{key}"] = value
            except Exception:
                # LLM evaluation is optional, don't fail if it errors
                pass

        # Clean and validate metrics
        return self._clean_metrics(result)

    def evaluate_code(self, code: str) -> Dict[str, float]:
        """
        Evaluate code synchronously using all available evaluators

        Args:
            code: Code to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self.evaluate_code_async(code))
        finally:
            loop.close()

    def _clean_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Clean and validate metrics, converting all values to floats

        Args:
            metrics: Raw metrics dictionary

        Returns:
            Cleaned metrics dictionary with float values
        """
        clean_metrics = {}

        for key, value in metrics.items():
            try:
                # Convert to float, handling various numeric types
                clean_metrics[key] = float(value)
            except (ValueError, TypeError):
                # If conversion fails, use 0.0 as default
                clean_metrics[key] = 0.0

        return clean_metrics

    def has_llm_evaluator(self) -> bool:
        """Check if LLM evaluator is available"""
        return self.llm_evaluator is not None

    def get_evaluation_summary(self, metrics: Dict[str, float]) -> str:
        """
        Generate a human-readable summary of evaluation metrics

        Args:
            metrics: Evaluation metrics

        Returns:
            Formatted summary string
        """
        if not metrics:
            return "No evaluation metrics available"

        lines = ["Evaluation Results:"]

        # Separate execution and LLM metrics
        exe_metrics = {k: v for k, v in metrics.items() if not k.startswith("llm_")}
        llm_metrics = {k: v for k, v in metrics.items() if k.startswith("llm_")}

        if exe_metrics:
            lines.append("  Execution Metrics:")
            for key, value in exe_metrics.items():
                lines.append(f"    {key}: {value:.4f}")

        if llm_metrics:
            lines.append("  LLM Metrics:")
            for key, value in llm_metrics.items():
                clean_key = key.replace("llm_", "")
                lines.append(f"    {clean_key}: {value:.4f}")

        return "\n".join(lines)
