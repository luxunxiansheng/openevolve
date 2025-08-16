"""
Program Evolution Environment for OpenContext

A stateless gymnasium environment that acts as a proxy for LLM-based program evolution.
Combines generic evolution instructions with specific action-provided instructions.
"""

from typing import Any, Dict, Optional, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from opencontext.common.actions import EvolutionAction
from opencontext.llm.llm_interface import LLMInterface
from opencontext.environment.evaluators import ExecutionEvaluator, LLMEvaluator
from opencontext.environment.program_evolution import ProgramEvolutionEngine


class ProgramEvolutionEnv(gym.Env):
    """Stateless gymnasium environment for LLM-based program evolution"""

    def __init__(
        self,
        llm: LLMInterface,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        reward_extractor: Optional[Callable[[Dict[str, Any]], float]] = None,
        language: str = "python",
    ):
        super().__init__()
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language
        self.reward_extractor = reward_extractor or self._default_reward_calculator
        self.evolution_engine = ProgramEvolutionEngine(llm, language)

        # Minimal gym spaces for compatibility
        self.action_space = spaces.Dict({"instruction": spaces.Text(max_length=10000)})
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

    def _default_reward_calculator(self, metrics: Dict[str, Any]) -> float:
        """Calculate reward from metrics using safe numeric average"""
        if not metrics:
            return 0.0

        numeric_values = []
        for value in metrics.values():
            if isinstance(value, (int, float)):
                try:
                    float_val = float(value)
                    if float_val == float_val:  # Check for not NaN
                        numeric_values.append(float_val)
                except (ValueError, TypeError, OverflowError):
                    continue

        return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0

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

    def step(self, action: EvolutionAction):
        """Execute evolution step: generate and evaluate program"""
        self.total_steps += 1

        if not isinstance(action, EvolutionAction):
            return self._create_error_response(
                f"Invalid action type: {type(action)}. Must be EvolutionAction."
            )

        try:
            program = self.evolution_engine.generate_code(action)
            if not program or not program.strip():
                return self._create_error_response("Empty program generated")

            metrics = self._evaluate_program(program)
            return self._create_success_response(program, metrics, action)

        except Exception as e:
            return self._create_error_response(f"Step failed: {str(e)}")

    def _create_success_response(
        self, program: str, metrics: Dict[str, float], action: EvolutionAction
    ):
        """Create successful step response"""
        obs = {
            "generated_program": program,
            "evaluation_metrics": self._metrics_to_array(metrics),
            "success": 1,
        }
        info = {
            "raw_metrics": metrics,
            "generation_success": True,
            "evaluation_success": True,
            "evolution_action": action.to_dict(),
        }
        reward = self.reward_extractor(metrics)
        return obs, reward, False, False, info

    def _create_error_response(self, error_msg: str):
        """Create error response"""
        obs = {
            "generated_program": "",
            "evaluation_metrics": np.zeros(10, dtype=np.float32),
            "success": 0,
        }
        info = {"error": error_msg, "raw_metrics": {"error": 1.0}}
        return obs, 0.0, False, False, info

    def _evaluate_program(self, code: str) -> Dict[str, float]:
        """Evaluate code using available evaluators"""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Execution evaluation
            exe_result = loop.run_until_complete(
                self.exe_evaluator.evaluate(code=code, language=self.language)
            )
            result = (
                exe_result.to_dict()
                if hasattr(exe_result, "to_dict")
                else exe_result.metrics if hasattr(exe_result, "metrics") else {}
            )

            # Optional LLM evaluation
            if self.llm_evaluator:
                try:
                    llm_result = loop.run_until_complete(
                        self.llm_evaluator.evaluate(code=code, language=self.language)
                    )
                    llm_metrics = (
                        llm_result.to_dict()
                        if hasattr(llm_result, "to_dict")
                        else llm_result.metrics if hasattr(llm_result, "metrics") else {}
                    )
                    for key, value in llm_metrics.items():
                        result[f"llm_{key}"] = value
                except Exception:
                    pass  # LLM evaluation is optional

            return {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in result.items()}
        finally:
            loop.close()

    def _metrics_to_array(self, metrics: Dict[str, float]) -> np.ndarray:
        """Convert metrics to fixed array"""
        array = np.zeros(10, dtype=np.float32)
        for i, value in enumerate(metrics.values()):
            if i >= 10:
                break
            array[i] = float(value)
        return array

    def render(self, mode="human"):
        """Simple render"""
        if mode == "human":
            print(f"Episode: {self.episode_count}, Steps: {self.total_steps}")

    def close(self):
        """Cleanup"""
        pass
