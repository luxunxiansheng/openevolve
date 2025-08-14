"""
Program Evolution Environment for OpenEvolve

A simple stateless gymnasium environment for evolving programs using LLMs and evaluation.
"""

import asyncio
from typing import Any, Dict, Optional, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from openevolve.llm.llm_interface import LLMInterface
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator


class ProgramEvolutionEnv(gym.Env):
    """
    Simple stateless gymnasium environment for program evolution.

    - Takes prompts as actions
    - Generates code with LLM
    - Evaluates code with evaluators
    - Returns raw metrics in info
    - User defines reward logic
    """

    def __init__(
        self,
        llm: LLMInterface,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        reward_extractor: Optional[Callable[[Dict[str, Any]], float]] = None,
        language: str = "python",
    ):
        super().__init__()

        self.llm = llm
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language
        self.reward_extractor = reward_extractor or self._default_reward

        # Simple action/observation spaces
        self.action_space = spaces.Text(max_length=50000)
        self.observation_space = spaces.Dict(
            {
                "generated_program": spaces.Text(max_length=50000),
                "evaluation_metrics": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                ),
                "success": spaces.Discrete(2),
            }
        )

        self.episode_count = 0
        self.total_steps = 0

    def reset(self, *, seed=None, options=None):
        """Reset for new episode"""
        super().reset(seed=seed)
        self.episode_count += 1

        obs = {
            "generated_program": "",
            "evaluation_metrics": np.zeros(10, dtype=np.float32),
            "success": 0,
        }
        info = {"episode": self.episode_count}
        return obs, info

    def step(self, action: str):
        """Execute one step: generate code and evaluate it"""
        self.total_steps += 1

        # Generate code
        try:
            new_program = self._generate_code(action)
            generation_ok = bool(new_program)
        except Exception as e:
            return self._error_response(f"Generation failed: {e}")

        # Evaluate code
        try:
            metrics = self._evaluate_code(new_program)
            evaluation_ok = bool(metrics)
        except Exception as e:
            return self._error_response(f"Evaluation failed: {e}", new_program)

        # Create response
        obs = {
            "generated_program": new_program,
            "evaluation_metrics": self._to_array(metrics),
            "success": int(generation_ok and evaluation_ok),
        }

        info = {
            "raw_metrics": metrics,
            "generation_success": generation_ok,
            "evaluation_success": evaluation_ok,
        }

        reward = self.reward_extractor(info)
        return obs, reward, False, False, info

    def _generate_code(self, prompt: str) -> str:
        """Generate code using LLM"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(self.llm.generate(prompt))

            # Handle ensemble response
            if isinstance(response, list):
                response = response[0] if response else ""

            # Extract code from response
            if "```" in str(response):
                start = response.find("```")
                end = response.find("```", start + 3)
                if end != -1:
                    code = response[start + 3 : end]
                    # Remove language identifier
                    lines = code.strip().split("\n")
                    if lines and lines[0].strip() in ["python", "py"]:
                        return "\n".join(lines[1:])
                    return code.strip()

            return str(response).strip()

        finally:
            loop.close()

    def _evaluate_code(self, code: str) -> Dict[str, float]:
        """Evaluate code using evaluators"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Execution evaluation
            result = loop.run_until_complete(
                self.exe_evaluator.evaluate(code=code, language=self.language)
            )

            # LLM evaluation (optional)
            if self.llm_evaluator:
                try:
                    llm_result = loop.run_until_complete(
                        self.llm_evaluator.evaluate(code=code, language=self.language)
                    )
                    # Add LLM metrics with prefix
                    for key, value in llm_result.items():
                        result[f"llm_{key}"] = value
                except Exception:
                    pass  # LLM evaluation is optional

            # Clean metrics
            clean_metrics = {}
            for key, value in result.items():
                try:
                    clean_metrics[key] = float(value)
                except (ValueError, TypeError):
                    clean_metrics[key] = 0.0

            return clean_metrics

        finally:
            loop.close()

    def _error_response(self, error_msg: str, program: str = ""):
        """Create error response"""
        obs = {
            "generated_program": program,
            "evaluation_metrics": np.zeros(10, dtype=np.float32),
            "success": 0,
        }
        info = {
            "error": error_msg,
            "raw_metrics": {"error": 1.0},
            "generation_success": bool(program),
            "evaluation_success": False,
        }
        reward = self.reward_extractor(info)
        return obs, reward, False, False, info

    def _to_array(self, metrics: Dict[str, float]) -> np.ndarray:
        """Convert metrics to fixed array"""
        array = np.zeros(10, dtype=np.float32)
        for i, value in enumerate(metrics.values()):
            if i >= 10:
                break
            array[i] = float(value)
        return array

    def _default_reward(self, info: Dict[str, Any]) -> float:
        """Default reward: average of metrics"""
        metrics = info.get("raw_metrics", {})
        return safe_numeric_average(metrics)

    def render(self, mode="human"):
        """Simple render"""
        if mode == "human":
            print(f"Episode: {self.episode_count}, Steps: {self.total_steps}")

    def close(self):
        """Cleanup"""
        pass

    @staticmethod
    def create_prompt(instruction: str, **context) -> str:
        """Helper to create prompts with context"""
        parts = [instruction]

        if "current_program" in context:
            parts.append(f"\nCurrent Program:\n```\n{context['current_program']}\n```")
        if "current_score" in context:
            parts.append(f"\nCurrent Score: {context['current_score']}")
        if "parent_program" in context:
            parts.append(f"\nParent Program:\n```\n{context['parent_program']}\n```")

        return "\n".join(parts)
