"""
Program Evolution Environment for OpenEvolve

A stateless gymnasium environment for evolving programs using LLMs and evaluation.
All context and history is provided through prompts rather than maintained internally.
"""

import asyncio
import logging
import time
import re
from typing import Any, Dict, Optional, Tuple, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from openevolve.llm.llm_interface import LLMInterface
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator

logger = logging.getLogger(__name__)


class ProgramEvolutionEnv(gym.Env):
    """
    Stateless gymnasium environment for program evolution.

    This environment doesn't maintain state between steps. Instead:
    - All necessary context (history, metrics, programs) is provided via the action prompt
    - The observation only contains the immediate result of the last action
    - Reward is calculated using a user-provided reward extraction function

    Action Space: Complete prompt containing all context and instructions
    Observation Space: Current program and evaluation results
    Reward: Calculated using user-provided reward extraction function
    """

    def __init__(
        self,
        llm: LLMInterface,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        reward_extractor: Optional[Callable[[Dict[str, Any]], float]] = None,
        language: str = "python",
        max_prompt_length: int = 50000,
        max_program_length: int = 50000,
    ):
        """
        Initialize the Stateless Program Evolution Environment

        Args:
            llm: Language model for program generation
            exe_evaluator: Execution evaluator for program assessment
            llm_evaluator: Optional LLM evaluator for additional assessment
            reward_extractor: Function that takes info dict and returns reward
            language: Programming language (default: python)
            max_prompt_length: Maximum length of comprehensive prompt actions
            max_program_length: Maximum length of program code
        """
        super().__init__()

        # Core components
        self.llm = llm
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language
        self.reward_extractor = reward_extractor or self._default_reward_extractor

        # Action space: Comprehensive prompts containing all context
        self.action_space = spaces.Text(max_length=max_prompt_length)

        # Observation space: Immediate results only (no persistent state)
        self.observation_space = spaces.Dict(
            {
                "generated_program": spaces.Text(max_length=max_program_length),
                "evaluation_metrics": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                ),
                "has_errors": spaces.Discrete(2),
                "generation_success": spaces.Discrete(2),
            }
        )

        # Episode tracking (minimal)
        self.episode_count = 0
        self.total_steps = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        self.episode_count += 1

        logger.info(f"Environment reset for new episode {self.episode_count} (stateless mode)")

        observation = self._get_empty_observation()
        info = {"episode": self.episode_count, "mode": "stateless"}

        return observation, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment using a comprehensive prompt"""
        self.total_steps += 1
        step_start_time = time.time()

        # Generate new program
        new_program = self._generate_program(action)
        generation_success = bool(new_program and not new_program.startswith("# Error"))

        if not generation_success:
            return self._handle_generation_error(new_program, step_start_time)

        # Evaluate the generated program
        raw_metrics = self._evaluate_program(new_program)
        evaluation_success = bool(raw_metrics)

        if not evaluation_success:
            return self._handle_evaluation_error(new_program, step_start_time)

        # Create successful observation
        observation = {
            "generated_program": new_program,
            "evaluation_metrics": self._metrics_to_array(raw_metrics),
            "has_errors": int("error" in raw_metrics and raw_metrics.get("error", 0) > 0),
            "generation_success": 1,
        }

        # Info dictionary with raw evaluation results
        info = {
            "raw_metrics": raw_metrics,
            "step_time": time.time() - step_start_time,
            "generation_success": True,
            "evaluation_success": True,
        }

        # Calculate reward using the provided extractor function
        reward = self.reward_extractor(info)

        current_score = safe_numeric_average(raw_metrics)
        logger.info(
            f"Step {self.total_steps}: reward={reward:.4f}, "
            f"score={current_score:.4f}, "
            f"success=True"
        )

        return observation, reward, False, False, info

    def _handle_generation_error(self, error_program: str, step_start_time: float):
        """Handle generation errors without exceptions"""
        observation = self._get_empty_observation()
        info = {
            "error": error_program or "Generation failed",
            "step": "generation",
            "step_time": time.time() - step_start_time,
            "generation_success": False,
            "evaluation_success": False,
            "raw_metrics": {},
        }
        reward = self.reward_extractor(info)
        return observation, reward, False, False, info

    def _handle_evaluation_error(self, new_program: str, step_start_time: float):
        """Handle evaluation errors without exceptions"""
        observation = {
            "generated_program": new_program,
            "evaluation_metrics": np.zeros(10, dtype=np.float32),
            "has_errors": 1,
            "generation_success": 1,
        }
        info = {
            "error": "Evaluation failed",
            "step": "evaluation",
            "step_time": time.time() - step_start_time,
            "generation_success": True,
            "evaluation_success": False,
            "raw_metrics": {"error": 1.0, "score": 0.0},
        }
        reward = self.reward_extractor(info)
        return observation, reward, False, False, info

    def _default_reward_extractor(self, info: Dict[str, Any]) -> float:
        """Default reward extractor that returns the average of raw metrics"""
        raw_metrics = info.get("raw_metrics", {})
        return safe_numeric_average(raw_metrics)

    def _get_empty_observation(self) -> Dict[str, Any]:
        """Get an empty observation for error cases"""
        return {
            "generated_program": "",
            "evaluation_metrics": np.zeros(10, dtype=np.float32),
            "has_errors": 1,
            "generation_success": 0,
        }

    def _generate_program(self, prompt: str) -> str:
        """Generate new program using LLM with the given prompt"""
        # Get or create event loop
        loop = self._get_event_loop()
        if not loop:
            return "# Error: Could not create event loop"

        # Call LLM
        response = self._run_async_safe(loop, self.llm.generate(prompt))
        if not response:
            return "# Error: No response from LLM"

        # Handle ensemble LLM returning a list
        if isinstance(response, list):
            response = response[0] if response else ""

        # Extract code from response
        return self._extract_code_from_response(response)

    def _evaluate_program(self, program: str) -> Dict[str, float]:
        """Evaluate program using evaluators"""
        # Get or create event loop
        loop = self._get_event_loop()
        if not loop:
            return {"error": 1.0, "score": 0.0}

        # Use execution evaluator
        result = self._run_async_safe(
            loop, self.exe_evaluator.evaluate(code=program, language=self.language)
        )

        if not result or not isinstance(result, dict):
            return {"error": 1.0, "score": 0.0}

        # Add LLM evaluator results if available
        if self.llm_evaluator:
            llm_result = self._run_async_safe(
                loop, self.llm_evaluator.evaluate(code=program, language=self.language)
            )
            if llm_result and isinstance(llm_result, dict):
                # Combine results
                for key, value in llm_result.items():
                    if key in result:
                        result[f"llm_{key}"] = value
                    else:
                        result[key] = value

        # Clean and convert metrics
        return self._clean_metrics(result)

    def _get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get or create event loop safely"""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop
            except Exception:
                return None

    def _run_async_safe(self, loop: asyncio.AbstractEventLoop, coro) -> Any:
        """Run async coroutine safely, return None on error"""
        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            return None

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response"""
        if not isinstance(response, str):
            return str(response).strip() if response else "# No response"

        # Extract code block if present
        if "```" in response:
            code_start = response.find("```")
            code_end = response.find("```", code_start + 3)
            if code_end != -1:
                code_block = response[code_start + 3 : code_end]
                lines = code_block.strip().split("\n")
                # Remove language identifier if present
                if lines and lines[0].strip() in ["python", "py", "java", "cpp", "c++"]:
                    return "\n".join(lines[1:])
                return code_block.strip()

        return response.strip()

    def _clean_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert metrics to clean float dict"""
        clean_metrics = {}
        for key, value in metrics.items():
            try:
                clean_metrics[key] = float(value)
            except (ValueError, TypeError):
                # Default values for non-numeric data
                if key == "error" or "error" in key.lower():
                    clean_metrics[key] = 1.0
                else:
                    clean_metrics[key] = 0.0
        return clean_metrics

    def _metrics_to_array(self, metrics: Dict[str, float]) -> np.ndarray:
        """Convert metrics dict to fixed-size array"""
        array = np.zeros(10, dtype=np.float32)

        if not metrics:
            return array

        # Fill array with metric values (up to 10)
        for i, (key, value) in enumerate(metrics.items()):
            if i >= 10:
                break
            array[i] = float(value) if isinstance(value, (int, float)) else 0.0

        return array

    def render(self, mode="human"):
        """Render the current state (stateless mode)"""
        if mode == "human":
            print(f"\n=== Program Evolution Environment (Stateless) ===")
            print(f"Episode: {self.episode_count}, Total Steps: {self.total_steps}")
            print("Mode: Stateless - no persistent state maintained")
            print("Context provided through action prompts")
            print("=" * 50)

    def close(self):
        """Clean up resources"""
        logger.info(f"Environment closed after {self.total_steps} total steps")

    @staticmethod
    def create_context_prompt(
        base_instruction: str,
        current_program: Optional[str] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        parent_program: Optional[str] = None,
        parent_metrics: Optional[Dict[str, float]] = None,
        history: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """Helper method to create a comprehensive context prompt"""
        prompt_parts = [base_instruction]

        # Add current program context
        if current_program:
            prompt_parts.append(f"\n## Current Program:\n```\n{current_program}\n```")

        if current_metrics:
            current_score = safe_numeric_average(current_metrics)
            prompt_parts.append(f"\nCurrent Score: {current_score:.4f}")
            metrics_str = "\n".join([f"- {k}: {v}" for k, v in current_metrics.items()])
            prompt_parts.append(f"\nCurrent Metrics:\n{metrics_str}")

        # Add parent context
        if parent_program:
            prompt_parts.append(f"\n## Parent Program:\n```\n{parent_program}\n```")

        if parent_metrics:
            parent_score = safe_numeric_average(parent_metrics)
            prompt_parts.append(f"\nPrevious Score: {parent_score:.4f}")

        # Add history and additional context
        if history:
            prompt_parts.append(f"\n## Evolution History:\n{history}")

        if additional_context:
            prompt_parts.append(f"\n## Additional Context:\n{additional_context}")

        return "\n".join(prompt_parts)
