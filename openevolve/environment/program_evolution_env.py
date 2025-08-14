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
        """
        Execute one step: process action, build prompts, generate and evaluate code

        Args:
            action: Evolution action in various formats (string, dict, or EvolutionAction)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.total_steps += 1

        # Process action
        evolution_action = self._process_action(action)
        if evolution_action is None:
            return self._error_response("Invalid action provided")

        # Build prompts
        prompts = self._build_prompts(evolution_action)
        if prompts is None:
            return self._error_response("Failed to build prompts")

        system_prompt, user_prompt = prompts

        # Generate code
        generation_result = self._generate_code(system_prompt, user_prompt)
        if not generation_result["success"]:
            return self._error_response(generation_result["error"])

        new_program = generation_result["program"]

        # Evaluate code
        evaluation_result = self._evaluate_code_safe(new_program)
        if not evaluation_result["success"]:
            return self._error_response(evaluation_result["error"], new_program)

        metrics = evaluation_result["metrics"]

        # Build response
        return self._build_response(
            evolution_action=evolution_action,
            new_program=new_program,
            metrics=metrics,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            generation_success=True,
            evaluation_success=True,
        )

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

    def _process_action(
        self, action: Union[str, Dict[str, Any], EvolutionAction]
    ) -> Optional[EvolutionAction]:
        """
        Process action into EvolutionAction format

        Returns:
            EvolutionAction if successful, None if failed
        """
        try:
            return self._parse_action(action)
        except (ValueError, TypeError, KeyError):
            return None

    def _build_prompts(self, evolution_action: EvolutionAction) -> Optional[tuple[str, str]]:
        """
        Build system and user prompts

        Returns:
            (system_prompt, user_prompt) tuple if successful, None if failed
        """
        try:
            return self.prompt_builder.build_prompts(evolution_action)
        except Exception:
            return None

    def _generate_code(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate code using LLM

        Returns:
            Result dict with success, error, and program fields
        """
        try:
            program = self.code_generator.generate_code(system_prompt, user_prompt)
            if not program or not program.strip():
                return {"success": False, "error": "Empty program generated", "program": ""}
            return {"success": True, "error": "", "program": program}
        except Exception as e:
            return {"success": False, "error": f"Code generation failed: {str(e)}", "program": ""}

    def _evaluate_code_safe(self, code: str) -> Dict[str, Any]:
        """
        Safely evaluate code

        Returns:
            Result dict with success, error, and metrics fields
        """
        try:
            metrics = self._evaluate_code(code)
            if not metrics:
                return {"success": False, "error": "No evaluation metrics returned", "metrics": {}}
            return {"success": True, "error": "", "metrics": metrics}
        except Exception as e:
            return {"success": False, "error": f"Code evaluation failed: {str(e)}", "metrics": {}}

    def _build_response(
        self,
        evolution_action: EvolutionAction,
        new_program: str,
        metrics: Dict[str, float],
        system_prompt: str,
        user_prompt: str,
        generation_success: bool,
        evaluation_success: bool,
    ) -> tuple:
        """
        Build the final step response

        Returns:
            Standard gym step tuple (obs, reward, terminated, truncated, info)
        """
        obs = {
            "generated_program": new_program,
            "evaluation_metrics": self._to_array(metrics),
            "success": int(generation_success and evaluation_success),
        }

        info = {
            "raw_metrics": metrics,
            "generation_success": generation_success,
            "evaluation_success": evaluation_success,
            "evolution_action": evolution_action.to_dict(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        reward = self.reward_extractor(info)
        return obs, reward, False, False, info

    def _evaluate_code(self, code: str) -> Dict[str, float]:
        """
        Evaluate code using available evaluators

        Args:
            code: Code to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        import asyncio

        # Start with execution evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get execution metrics
            result = loop.run_until_complete(
                self.exe_evaluator.evaluate(code=code, language=self.language)
            )

            # Add LLM evaluation if available
            if self.llm_evaluator:
                try:
                    llm_result = loop.run_until_complete(
                        self.llm_evaluator.evaluate(code=code, language=self.language)
                    )
                    # Add LLM metrics with prefix to avoid conflicts
                    for key, value in llm_result.items():
                        result[f"llm_{key}"] = value
                except Exception:
                    # LLM evaluation is optional, don't fail if it errors
                    pass

            # Clean and validate metrics
            return self._clean_metrics(result)
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
