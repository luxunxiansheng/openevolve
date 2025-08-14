"""
Program Evolution Environment for OpenEvolve

A stateless gymnasium environment that acts as a proxy for LLM-based program evolution.
Combines generic evolution instructions with specific action-provided instructions.
"""

from typing import Any, Dict, Optional, Callable, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from openevolve.common.actions import EvolutionAction
from openevolve.llm.llm_interface import LLMInterface
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator
from openevolve.environment.prompts import PromptBuilder
from openevolve.environment.code_generation import CodeGenerator
from openevolve.environment.evaluation import CodeEvaluator


class ProgramEvolutionEnv(gym.Env):
    """
    Stateless gymnasium environment acting as a proxy for LLM-based program evolution.

    - Acts as proxy between actions and LLM backend
    - Combines generic evolution instructions with action-specific instructions
    - Handles LLM communication and response parsing
    - Evaluates generated code and returns metrics
    """

    def __init__(
        self,
        llm: LLMInterface,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        reward_extractor: Optional[Callable[[Dict[str, Any]], float]] = None,
        language: str = "python",
        max_code_length: int = 20480,
        default_mode: str = "full_rewrite",
    ):
        super().__init__()

        self.llm = llm
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language
        self.reward_extractor = reward_extractor or self._default_reward
        self.max_code_length = max_code_length
        self.default_mode = default_mode

        # Initialize specialized components
        self.prompt_builder = PromptBuilder(language)
        self.code_generator = CodeGenerator(llm, language, max_code_length)
        self.code_evaluator = CodeEvaluator(exe_evaluator, llm_evaluator, language)

        # Action space supports structured actions
        self.action_space = spaces.Dict(
            {
                "instruction": spaces.Text(max_length=10000),
                "current_program": spaces.Text(max_length=50000),
                "current_score": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "parent_program": spaces.Text(max_length=50000),
                "previous_attempts": spaces.Text(max_length=10000),
                "context": spaces.Text(max_length=10000),
                "mode": spaces.Text(max_length=20),
            }
        )

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

    def step(self, action: Union[str, Dict[str, Any], EvolutionAction]):
        """Execute one step: process action, build prompts, generate and evaluate code"""
        self.total_steps += 1

        # Process the action
        try:
            evolution_action = self._parse_action(action)
            system_prompt, user_prompt = self.prompt_builder.build_prompts(evolution_action)
        except Exception as e:
            return self._error_response(f"Action processing failed: {e}")

        # Generate code via LLM proxy
        try:
            new_program = self.code_generator.generate_code(system_prompt, user_prompt)
            generation_ok = bool(new_program)
        except Exception as e:
            return self._error_response(f"Generation failed: {e}")

        # Evaluate code
        try:
            metrics = self.code_evaluator.evaluate_code(new_program)
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
            "evolution_action": evolution_action.to_dict(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        reward = self.reward_extractor(info)
        return obs, reward, False, False, info

    def _parse_action(self, action: Union[str, Dict[str, Any], EvolutionAction]) -> EvolutionAction:
        """Parse different action formats into EvolutionAction"""
        if isinstance(action, EvolutionAction):
            return action
        elif isinstance(action, dict):
            return EvolutionAction.from_dict(action)
        elif isinstance(action, str):
            # Fallback: treat string as instruction
            return EvolutionAction(instruction=action, mode=self.default_mode)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

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
