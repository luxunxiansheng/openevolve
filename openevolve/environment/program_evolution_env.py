"""
Program Evolution Environment for OpenEvolve

A gymnasium environment for evolving programs using LLMs and evaluation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from openevolve.llm.llm_interface import LLMInterface
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator

logger = logging.getLogger(__name__)


class ProgramEvolutionEnv(gym.Env):
    """
    Gymnasium environment for single program evolution.

    Action Space: Text prompt to guide program generation
    Observation Space: Current program state and metrics
    Reward: Improvement in program performance
    """

    def __init__(
        self,
        llm: LLMInterface,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        language: str = "python",
        max_prompt_length: int = 10000,
        max_program_length: int = 50000,
        reward_scale: float = 1.0,
        improvement_bonus: float = 0.1,
        diversity_bonus: float = 0.0,
        penalty_for_errors: float = -0.1,
    ):
        """
        Initialize the Program Evolution Environment

        Args:
            llm: Language model for program generation
            exe_evaluator: Execution evaluator for program assessment
            llm_evaluator: Optional LLM evaluator for additional assessment
            language: Programming language (default: python)
            max_prompt_length: Maximum length of prompt actions
            max_program_length: Maximum length of program code
            reward_scale: Scale factor for rewards
            improvement_bonus: Bonus for any improvement
            diversity_bonus: Bonus for diverse solutions
            penalty_for_errors: Penalty for syntax/runtime errors
        """
        super().__init__()

        # Core components
        self.llm = llm
        self.exe_evaluator = exe_evaluator
        self.llm_evaluator = llm_evaluator
        self.language = language

        # Reward configuration
        self.reward_scale = reward_scale
        self.improvement_bonus = improvement_bonus
        self.diversity_bonus = diversity_bonus
        self.penalty_for_errors = penalty_for_errors

        # Action space: Text prompts
        self.action_space = spaces.Text(max_length=max_prompt_length)

        # Observation space: Program state and metrics
        self.observation_space = spaces.Dict(
            {
                "current_program": spaces.Text(max_length=max_program_length),
                "parent_program": spaces.Text(max_length=max_program_length),
                "current_metrics": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                ),
                "parent_metrics": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                ),
                "iteration": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
                "has_errors": spaces.Discrete(2),  # 0: no errors, 1: has errors
            }
        )

        # Internal state
        self.current_program: Optional[str] = None
        self.parent_program: Optional[str] = None
        self.current_metrics: Dict[str, float] = {}
        self.parent_metrics: Dict[str, float] = {}
        self.iteration = 0
        self.has_errors = False

        # Episode tracking
        self.episode_count = 0
        self.total_iterations = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment with an initial program

        Args:
            seed: Random seed (not used currently)
            options: Reset options, can contain 'initial_program'

        Returns:
            observation: Current state observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Get initial program from options or use default
        if options and "initial_program" in options:
            initial_program = options["initial_program"]
        else:
            initial_program = self._get_default_initial_program()

        # Evaluate initial program
        initial_metrics = self._evaluate_program(initial_program)

        # Set initial state
        self.current_program = initial_program
        self.parent_program = ""
        self.current_metrics = initial_metrics
        self.parent_metrics = {}
        self.iteration = 0
        self.has_errors = "error" in initial_metrics

        # Increment episode counter
        self.episode_count += 1

        logger.info(f"Environment reset with initial program (episode {self.episode_count})")

        observation = self._get_observation()
        info = {"initial_metrics": initial_metrics}

        return observation, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment

        Args:
            action: Text prompt for program generation

        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.iteration += 1
        self.total_iterations += 1

        # Store current state as parent
        self.parent_program = self.current_program
        self.parent_metrics = self.current_metrics.copy()

        # Generate new program using LLM
        try:
            new_program = self._generate_program(action)
            logger.debug(f"Generated new program (length: {len(new_program)})")
        except Exception as e:
            logger.error(f"Error generating program: {e}")
            # Return current state with penalty
            observation = self._get_observation()
            reward = self.penalty_for_errors
            info = {"error": str(e), "step": "generation"}
            return observation, reward, False, False, info

        # Check if generation returned an error string
        if new_program.startswith("# Error generating code:"):
            logger.error("LLM generation returned error")
            observation = self._get_observation()
            reward = self.penalty_for_errors
            info = {"error": new_program, "step": "generation"}
            return observation, reward, False, False, info

        # Evaluate new program
        try:
            new_metrics = self._evaluate_program(new_program)
            logger.debug(f"Evaluated program with metrics: {new_metrics}")
        except Exception as e:
            logger.error(f"Error evaluating program: {e}")
            # Return current state with penalty
            observation = self._get_observation()
            reward = self.penalty_for_errors
            info = {"error": str(e), "step": "evaluation"}
            return observation, reward, False, False, info

        # Update state
        self.current_program = new_program
        self.current_metrics = new_metrics
        self.has_errors = "error" in new_metrics

        # Calculate reward
        reward = self._calculate_reward(new_metrics, self.parent_metrics)

        # Get new observation
        observation = self._get_observation()

        # Episode termination (could be based on various criteria)
        terminated = False  # Single step episodes for now
        truncated = False

        # Info dictionary
        info = {
            "new_metrics": new_metrics,
            "parent_metrics": self.parent_metrics,
            "reward_components": self._get_reward_components(new_metrics, self.parent_metrics),
            "iteration": self.iteration,
            "action_length": len(action),
        }

        logger.info(
            f"Step {self.iteration}: reward={reward:.4f}, "
            f"current_score={safe_numeric_average(new_metrics):.4f}"
        )

        return observation, reward, terminated, truncated, info

    def _generate_program(self, prompt: str) -> str:
        """Generate new program using LLM with the given prompt"""
        # Run async LLM call in sync context
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            response = loop.run_until_complete(self.llm.generate(prompt))

            # Handle ensemble LLM returning a list
            if isinstance(response, list):
                response = response[0] if response else ""

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"# Error generating code: {e}"

        # Extract code from response (simple heuristic)
        if isinstance(response, str) and "```" in response:
            # Extract code block
            code_start = response.find("```")
            code_end = response.find("```", code_start + 3)
            if code_end != -1:
                code_block = response[code_start + 3 : code_end]
                # Remove language identifier if present
                lines = code_block.strip().split("\n")
                if lines and lines[0].strip() in ["python", "py", "java", "cpp", "c++"]:
                    return "\n".join(lines[1:])
                return code_block.strip()

        return str(response).strip() if response else "# No response from LLM"

    def _evaluate_program(self, program: str) -> Dict[str, float]:
        """Evaluate program using evaluators"""
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Use execution evaluator
            result = loop.run_until_complete(
                self.exe_evaluator.evaluate(code=program, language=self.language)
            )

            # If we have an LLM evaluator, combine results
            if self.llm_evaluator:
                try:
                    llm_result = loop.run_until_complete(
                        self.llm_evaluator.evaluate(code=program, language=self.language)
                    )
                    # Combine results (simple averaging for overlapping keys)
                    for key, value in llm_result.items():
                        if key in result:
                            result[f"llm_{key}"] = value
                        else:
                            result[key] = value
                except Exception as e:
                    logger.warning(f"LLM evaluation failed: {e}")

            # Ensure we always have numeric metrics
            if not result or not isinstance(result, dict):
                return {"error": 1.0, "score": 0.0}

            # Convert all values to float where possible
            clean_metrics = {}
            for key, value in result.items():
                try:
                    clean_metrics[key] = float(value)
                except (ValueError, TypeError):
                    if key == "error" or "error" in key.lower():
                        clean_metrics[key] = 1.0
                    else:
                        clean_metrics[key] = 0.0

            return clean_metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": 1.0, "score": 0.0}

    def _calculate_reward(
        self, current_metrics: Dict[str, float], parent_metrics: Dict[str, float]
    ) -> float:
        """Calculate reward based on improvement"""
        if not parent_metrics:
            # First evaluation - return current score
            return safe_numeric_average(current_metrics) * self.reward_scale

        # Calculate improvement
        current_score = safe_numeric_average(current_metrics)
        parent_score = safe_numeric_average(parent_metrics)
        improvement = current_score - parent_score

        # Base reward is improvement
        reward = improvement * self.reward_scale

        # Add improvement bonus for any positive change
        if improvement > 0:
            reward += self.improvement_bonus

        # Add penalty for errors
        if current_metrics.get("error", 0) > 0:
            reward += self.penalty_for_errors

        return reward

    def _get_reward_components(
        self, current_metrics: Dict[str, float], parent_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Get detailed breakdown of reward components"""
        if not parent_metrics:
            return {"initial_score": safe_numeric_average(current_metrics)}

        current_score = safe_numeric_average(current_metrics)
        parent_score = safe_numeric_average(parent_metrics)
        improvement = current_score - parent_score

        components = {
            "improvement": improvement * self.reward_scale,
            "current_score": current_score,
            "parent_score": parent_score,
        }

        if improvement > 0:
            components["improvement_bonus"] = self.improvement_bonus

        if current_metrics.get("error", 0) > 0:
            components["error_penalty"] = self.penalty_for_errors

        return components

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        # Pad metrics to fixed size (10 elements)
        current_metrics_array = self._metrics_to_array(self.current_metrics)
        parent_metrics_array = self._metrics_to_array(self.parent_metrics)

        return {
            "current_program": self.current_program or "",
            "parent_program": self.parent_program or "",
            "current_metrics": current_metrics_array,
            "parent_metrics": parent_metrics_array,
            "iteration": np.array([self.iteration], dtype=np.int32),
            "has_errors": int(self.has_errors),
        }

    def _metrics_to_array(self, metrics: Dict[str, float]) -> np.ndarray:
        """Convert metrics dict to fixed-size array"""
        # Create fixed-size array (10 elements)
        array = np.zeros(10, dtype=np.float32)

        if not metrics:
            return array

        # Fill array with metric values (up to 10)
        for i, (key, value) in enumerate(metrics.items()):
            if i >= 10:
                break
            try:
                array[i] = float(value)
            except (ValueError, TypeError):
                array[i] = 0.0

        return array

    def _get_default_initial_program(self) -> str:
        """Get default initial program if none provided"""
        if self.language == "python":
            return '''def hello_world():
    """A simple hello world function"""
    return "Hello, World!"'''
        else:
            return "// Initial program"

    def render(self, mode="human"):
        """Render the current state"""
        if mode == "human":
            print(f"\n=== Program Evolution Environment ===")
            print(f"Episode: {self.episode_count}, Iteration: {self.iteration}")
            print(f"Current Score: {safe_numeric_average(self.current_metrics):.4f}")
            if self.parent_metrics:
                print(f"Parent Score: {safe_numeric_average(self.parent_metrics):.4f}")
            print(f"Has Errors: {self.has_errors}")
            print(f"Program Length: {len(self.current_program or '')}")
            print("=" * 40)

    def close(self):
        """Clean up resources"""
        logger.info(f"Environment closed after {self.total_iterations} total iterations")
