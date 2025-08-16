"""
Program Evolution Environment for OpenContext

A stateless gymnasium environment that acts as a proxy for LLM-based program evolution.
Combines generic evolution instructions with specific action-provided instructions.
"""

from typing import Any, Dict, Optional, Callable
import asyncio
import logging
import time
from dataclasses import asdict
import gymnasium as gym
from gymnasium import spaces

from opencontext.common.actions import EvolutionAction
from opencontext.llm.llm_interface import LLMInterface
from opencontext.environment.program_evaluation import ExecutionEvaluator, LLMEvaluator
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
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language
        self.reward_extractor = reward_extractor or self._default_reward_calculator
        self.logger = logger or logging.getLogger(__name__)
        self.evolution_engine = ProgramEvolutionEngine(llm, language, logger=self.logger)

        # Minimal gym spaces for compatibility
        self.action_space = spaces.Dict({"instruction": spaces.Text(max_length=10000)})
        self.observation_space = spaces.Dict(
            {
                "generated_program": spaces.Text(max_length=50000),
                "evaluation_metrics": spaces.Dict({}),  # Dynamic dict for flexible metrics
                "success": spaces.Discrete(2),
            }
        )

        self.episode_count = 0
        self.total_steps = 0

        self.logger.info(
            "ProgramEvolutionEnv initialized",
            extra={
                "language": language,
                "has_llm_evaluator": llm_evaluator is not None,
            },
        )

    def reset(self, *, seed=None, options=None):
        """Reset for new episode"""
        super().reset(seed=seed)
        self.episode_count += 1

        self.logger.info(
            "Environment reset", extra={"episode_count": self.episode_count, "seed": seed}
        )

        obs = {
            "generated_program": "",
            "evaluation_metrics": {},
            "success": 0,
        }
        info = {"episode": self.episode_count}
        return obs, info

    def step(self, action: EvolutionAction):
        """Execute evolution step: generate and evaluate program"""
        step_start_time = time.time()
        self.total_steps += 1

        self.logger.info(
            "Evolution step started",
            extra={
                "step": self.total_steps,
                "episode": self.episode_count,
                "action_goal": action.goal,
                "action_mode": action.mode.value,
            },
        )

        if not isinstance(action, EvolutionAction):
            error_msg = f"Invalid action type: {type(action)}. Must be EvolutionAction."
            self.logger.error("Invalid action type", extra={"action_type": str(type(action))})
            return self._create_error_response(error_msg)

        try:
            # Generate code using evolution engine
            generation_start = time.time()
            program = asyncio.run(self.evolution_engine.generate_code(action))
            generation_time = time.time() - generation_start

            if not program or not program.strip():
                self.logger.warning("Empty program generated")
                return self._create_error_response("Empty program generated")

            self.logger.debug(
                "Code generation completed",
                extra={
                    "program_length": len(program),
                    "generation_time_ms": int(generation_time * 1000),
                },
            )

            # Evaluate the generated program
            evaluation_start = time.time()
            metrics = self._evaluate_program(program)
            evaluation_time = time.time() - evaluation_start

            self.logger.info(
                "Evolution step completed successfully",
                extra={
                    "step": self.total_steps,
                    "metrics": metrics,
                    "generation_time_ms": int(generation_time * 1000),
                    "evaluation_time_ms": int(evaluation_time * 1000),
                    "total_time_ms": int((time.time() - step_start_time) * 1000),
                },
            )

            return self._create_success_response(program, metrics, action)

        except Exception as e:
            error_msg = f"Step failed: {str(e)}"
            self.logger.error(
                "Evolution step failed",
                extra={
                    "step": self.total_steps,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_time_ms": int((time.time() - step_start_time) * 1000),
                },
                exc_info=True,
            )
            return self._create_error_response(error_msg)

    def render(self, mode="human"):
        """Simple render"""
        if mode == "human":
            print(f"Episode: {self.episode_count}, Steps: {self.total_steps}")

    def close(self):
        """Cleanup"""
        pass

    def _create_success_response(
        self, program: str, metrics: Dict[str, float], action: EvolutionAction
    ):
        """Create successful step response"""
        obs = {
            "generated_program": program,
            "evaluation_metrics": metrics,  # Keep as dict
            "success": 1,
        }
        info = {
            "raw_metrics": metrics,
            "generation_success": True,
            "evaluation_success": True,
            "evolution_action": asdict(action),
        }
        reward = self.reward_extractor(metrics)
        return obs, reward, False, False, info

    def _create_error_response(self, error_msg: str):
        """Create error response"""
        obs = {
            "generated_program": "",
            "evaluation_metrics": {},
            "success": 0,
        }
        info = {"error": error_msg, "raw_metrics": {"error": 1.0}}
        return obs, 0.0, False, False, info

    def _evaluate_program(self, code: str) -> Dict[str, float]:
        """Evaluate code using available evaluators"""
        evaluation_start = time.time()

        self.logger.debug(
            "Starting program evaluation",
            extra={"code_length": len(code), "has_llm_evaluator": self.llm_evaluator is not None},
        )

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Execution evaluation
            exe_start = time.time()
            exe_result = loop.run_until_complete(
                self.exe_evaluator.evaluate(code=code, language=self.language)
            )
            exe_time = time.time() - exe_start

            result = (
                exe_result.to_dict()
                if hasattr(exe_result, "to_dict")
                else exe_result.metrics if hasattr(exe_result, "metrics") else {}
            )

            self.logger.debug(
                "Execution evaluation completed",
                extra={"exe_metrics": result, "exe_time_ms": int(exe_time * 1000)},
            )

            # Optional LLM evaluation
            if self.llm_evaluator:
                try:
                    llm_start = time.time()
                    llm_result = loop.run_until_complete(
                        self.llm_evaluator.evaluate(code=code, language=self.language)
                    )
                    llm_time = time.time() - llm_start

                    llm_metrics = (
                        llm_result.to_dict()
                        if hasattr(llm_result, "to_dict")
                        else llm_result.metrics if hasattr(llm_result, "metrics") else {}
                    )

                    for key, value in llm_metrics.items():
                        result[f"llm_{key}"] = value

                    self.logger.debug(
                        "LLM evaluation completed",
                        extra={"llm_metrics": llm_metrics, "llm_time_ms": int(llm_time * 1000)},
                    )

                except Exception as e:
                    self.logger.warning(
                        "LLM evaluation failed",
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )

            # Convert all values to float
            final_result = {
                k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in result.items()
            }

            self.logger.debug(
                "Program evaluation completed",
                extra={
                    "final_metrics": final_result,
                    "total_evaluation_time_ms": int((time.time() - evaluation_start) * 1000),
                },
            )

            return final_result

        finally:
            loop.close()

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
