"""
Program Evolution Environment for OpenEvolve

A stateless gymnasium environment for evolving programs using LLMs and evaluation.
All context and history is provided through prompts rather than maintained internally.
"""

import asyncio
import logging
import time
import re
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
    Stateless gymnasium environment for program evolution.

    This environment doesn't maintain state between steps. Instead:
    - All necessary context (history, metrics, programs) is provided via the action prompt
    - The observation only contains the immediate result of the last action
    - Reward is calculated based on information provided in the action

    Action Space: Complete prompt containing all context and instructions
    Observation Space: Current program and evaluation results
    Reward: Improvement based on metrics provided in the action
    """

    def __init__(
        self,
        llm: LLMInterface,
        exe_evaluator: ExecutionEvaluator,
        llm_evaluator: Optional[LLMEvaluator] = None,
        language: str = "python",
        max_prompt_length: int = 50000,  # Increased for comprehensive context
        max_program_length: int = 50000,
        reward_scale: float = 1.0,
        improvement_bonus: float = 0.1,
        diversity_bonus: float = 0.0,
        penalty_for_errors: float = -0.1,
    ):
        """
        Initialize the Stateless Program Evolution Environment

        Args:
            llm: Language model for program generation
            exe_evaluator: Execution evaluator for program assessment
            llm_evaluator: Optional LLM evaluator for additional assessment
            language: Programming language (default: python)
            max_prompt_length: Maximum length of comprehensive prompt actions
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

        # Action space: Comprehensive prompts containing all context
        self.action_space = spaces.Text(max_length=max_prompt_length)

        # Observation space: Immediate results only (no persistent state)
        self.observation_space = spaces.Dict(
            {
                "generated_program": spaces.Text(max_length=max_program_length),
                "evaluation_metrics": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                ),
                "has_errors": spaces.Discrete(2),  # 0: no errors, 1: has errors
                "generation_success": spaces.Discrete(2),  # 0: failed, 1: success
            }
        )

        # No persistent internal state - environment is stateless
        # All context must be provided through the action prompt

        # Episode tracking (minimal)
        self.episode_count = 0
        self.total_steps = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode

        Args:
            seed: Random seed (not used currently)
            options: Reset options (not used in stateless mode)

        Returns:
            observation: Empty initial observation (no state maintained)
            info: Episode information
        """
        super().reset(seed=seed)

        # Increment episode counter
        self.episode_count += 1

        logger.info(f"Environment reset for new episode {self.episode_count} (stateless mode)")

        # Return empty observation - no initial state
        observation = self._get_empty_observation()
        info = {"episode": self.episode_count, "mode": "stateless"}

        return observation, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment using a comprehensive prompt

        The action should be a complete prompt that contains:
        - Instructions for code generation/modification
        - Current program context and history
        - Previous metrics and performance data
        - Any other context needed for the LLM

        Args:
            action: Complete prompt containing all necessary context

        Returns:
            observation: New program and evaluation results
            reward: Reward calculated from metrics (if provided in context)
            terminated: Whether episode is finished (always False in stateless mode)
            truncated: Whether episode was truncated (always False)
            info: Additional information about the step
        """
        self.total_steps += 1

        step_start_time = time.time()

        # Parse context from the action prompt if available
        context = self._parse_action_context(action)

        # Generate new program using LLM with the complete prompt
        try:
            new_program = self._generate_program(action)
            logger.debug(f"Generated new program (length: {len(new_program)})")
            generation_success = True
        except Exception as e:
            logger.error(f"Error generating program: {e}")
            # Return empty observation with penalty
            observation = self._get_empty_observation()
            reward = self.penalty_for_errors
            info = {
                "error": str(e),
                "step": "generation",
                "step_time": time.time() - step_start_time,
                "generation_success": False,
            }
            return observation, reward, False, False, info

        # Check if generation returned an error string
        if new_program.startswith("# Error generating code:"):
            logger.error("LLM generation returned error")
            observation = self._get_empty_observation()
            reward = self.penalty_for_errors
            info = {
                "error": new_program,
                "step": "generation",
                "step_time": time.time() - step_start_time,
                "generation_success": False,
            }
            return observation, reward, False, False, info

        # Evaluate new program
        try:
            new_metrics = self._evaluate_program(new_program)
            logger.debug(f"Evaluated program with metrics: {new_metrics}")
            evaluation_success = True
        except Exception as e:
            logger.error(f"Error evaluating program: {e}")
            # Return observation with generated program but no metrics
            observation = {
                "generated_program": new_program,
                "evaluation_metrics": np.zeros(10, dtype=np.float32),
                "has_errors": 1,
                "generation_success": 1,  # Generation succeeded, evaluation failed
            }
            reward = self.penalty_for_errors
            info = {
                "error": str(e),
                "step": "evaluation",
                "step_time": time.time() - step_start_time,
                "generation_success": True,
                "evaluation_success": False,
                "metrics": {"error": 1.0, "score": 0.0},  # Add metrics for test compatibility
                "reward_calculation": {"error_only": self.penalty_for_errors},
                "context_parsed": context is not None,
            }
            return observation, reward, False, False, info

        # Calculate reward based on context and new metrics
        reward = self._calculate_reward_from_context(new_metrics, context)

        # Create observation with results
        observation = {
            "generated_program": new_program,
            "evaluation_metrics": self._metrics_to_array(new_metrics),
            "has_errors": int("error" in new_metrics and new_metrics.get("error", 0) > 0),
            "generation_success": int(generation_success),
        }

        # Episodes don't terminate in stateless mode - each step is independent
        terminated = False
        truncated = False

        # Info dictionary
        info = {
            "metrics": new_metrics,
            "reward_calculation": self._get_reward_explanation(new_metrics, context),
            "step_time": time.time() - step_start_time,
            "generation_success": generation_success,
            "evaluation_success": True,
            "context_parsed": context is not None,
        }

        current_score = safe_numeric_average(new_metrics)
        logger.info(
            f"Step {self.total_steps}: reward={reward:.4f}, "
            f"score={current_score:.4f}, "
            f"success={generation_success}"
        )

        return observation, reward, terminated, truncated, info

    def _parse_action_context(self, action: str) -> Optional[Dict[str, Any]]:
        """
        Parse context information from the action prompt

        Looks for structured context in the prompt like:
        - Current score: X.XX
        - Previous score: X.XX
        - Metrics: {...}

        Args:
            action: The complete prompt containing context

        Returns:
            Dictionary with parsed context or None if no context found
        """
        context = {}

        # Try to extract current score
        current_score_match = re.search(
            r"current[_\s]*score:?\s*([0-9]*\.?[0-9]+)", action, re.IGNORECASE
        )
        if current_score_match:
            context["current_score"] = float(current_score_match.group(1))

        # Try to extract previous/parent score
        parent_score_match = re.search(
            r"(?:previous|parent)[_\s]*score:?\s*([0-9]*\.?[0-9]+)", action, re.IGNORECASE
        )
        if parent_score_match:
            context["parent_score"] = float(parent_score_match.group(1))

        # Try to extract metrics section
        metrics_match = re.search(r"metrics:?\s*\{([^}]+)\}", action, re.IGNORECASE)
        if metrics_match:
            try:
                # Simple parsing of key: value pairs
                metrics_str = metrics_match.group(1)
                metrics = {}
                for line in metrics_str.split(","):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        try:
                            metrics[key.strip()] = float(value.strip())
                        except ValueError:
                            pass
                context["metrics"] = metrics
            except:
                pass

        return context if context else None

    def _get_empty_observation(self) -> Dict[str, Any]:
        """Get an empty observation for error cases"""
        return {
            "generated_program": "",
            "evaluation_metrics": np.zeros(10, dtype=np.float32),
            "has_errors": 1,
            "generation_success": 0,
        }

    def _calculate_reward_from_context(
        self, current_metrics: Dict[str, float], context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate reward based on current metrics and context from the prompt

        Args:
            current_metrics: Metrics from evaluating the generated program
            context: Context parsed from the action prompt

        Returns:
            Calculated reward
        """
        current_score = safe_numeric_average(current_metrics)

        # If we have context with previous score, calculate improvement
        if context and "parent_score" in context:
            parent_score = context["parent_score"]
            improvement = current_score - parent_score

            # Base reward is improvement
            reward = improvement * self.reward_scale

            # Add improvement bonus for any positive change
            if improvement > 0:
                reward += self.improvement_bonus

        elif context and "current_score" in context:
            # Use provided current score if available
            reward = context["current_score"] * self.reward_scale
        else:
            # No context - use current score as reward
            reward = current_score * self.reward_scale

        # Add penalty for errors
        if current_metrics.get("error", 0) > 0:
            reward += self.penalty_for_errors

        return reward

    def _get_reward_explanation(
        self, current_metrics: Dict[str, float], context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Get detailed explanation of reward calculation"""
        current_score = safe_numeric_average(current_metrics)
        explanation = {"current_score": current_score}

        if context and "parent_score" in context:
            parent_score = context["parent_score"]
            improvement = current_score - parent_score
            explanation.update(
                {
                    "parent_score": parent_score,
                    "improvement": improvement,
                    "improvement_reward": improvement * self.reward_scale,
                }
            )

            if improvement > 0:
                explanation["improvement_bonus"] = self.improvement_bonus

        if current_metrics.get("error", 0) > 0:
            explanation["error_penalty"] = self.penalty_for_errors

        return explanation

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
        """Calculate reward based on improvement (backward compatibility)"""
        if not parent_metrics:
            return safe_numeric_average(current_metrics) * self.reward_scale

        current_score = safe_numeric_average(current_metrics)
        parent_score = safe_numeric_average(parent_metrics)
        improvement = current_score - parent_score

        reward = improvement * self.reward_scale

        if improvement > 0:
            reward += self.improvement_bonus

        if current_metrics.get("error", 0) > 0:
            reward += self.penalty_for_errors

        return reward

    def _get_reward_components(
        self, current_metrics: Dict[str, float], parent_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Get detailed breakdown of reward components (backward compatibility)"""
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
        """Get current observation (deprecated - for backward compatibility)"""
        logger.warning("_get_observation called in stateless mode - returning empty observation")
        return self._get_empty_observation()

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
        """
        Helper method to create a comprehensive context prompt

        Args:
            base_instruction: Base instruction for the LLM
            current_program: Current program code
            current_metrics: Current program metrics
            parent_program: Parent program code
            parent_metrics: Parent program metrics
            history: Evolution history text
            additional_context: Any additional context

        Returns:
            Complete prompt with embedded context
        """
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
