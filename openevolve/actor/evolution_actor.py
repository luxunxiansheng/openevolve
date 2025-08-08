import logging
import time
import uuid

import ray
from ray.actor import ActorHandle

from openevolve.actor.actor import Actor, ActionResult
from openevolve.actor.config import EvolutionActorConfig
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.critic.llm_critic import LLMCritic
from openevolve.llm.llm_interface import LLMInterface
from openevolve.prompt.sampler import PromptSampler

from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class EvolutionActor(Actor):
    def __init__(
        self,
        database: ActorHandle,  # Explicit Ray actor handle
        prompt_sampler: PromptSampler,
        llm_actor_client: LLMInterface,
        llm_critic: LLMCritic,
        exe_critic: PythonExecutionCritic,
        config: EvolutionActorConfig = EvolutionActorConfig(),
    ) -> None:
        self.database = database
        self.prompt_sampler = prompt_sampler
        self.llm_actor_client = llm_actor_client
        self.llm_critic = llm_critic
        self.exe_critic = exe_critic
        self.language = config.language
        self.diff_based_evolution = config.diff_based_evolution
        self.max_code_length = config.max_code_length
        self.artifacts = {}
        self.use_llm_critic = config.use_llm_critic
        self.llm_feedback_weight = config.llm_feedback_weight
        self.artifacts_enabled = config.artifacts_enabled
        self.island_top_programs_limit = config.island_top_programs_limit  # Limit for top programs per island
        self.island_previous_programs_limit = config.island_previous_programs_limit  # Limit for previous programs per island

    def _enclose_code_block(self, code: str) -> str:
        """
        Enclose code block with EVOLVE-BLOCK-START and EVOLVE-BLOCK-END comments if not already present.
        """
        if "# EVOLVE-BLOCK-START" not in code and "# EVOLVE-BLOCK-END" not in code:
            return f"# EVOLVE-BLOCK-START\n{code}\n# EVOLVE-BLOCK-END"
        return code

    async def act(self, **kwargs) -> ActionResult:
        """
        Perform the evolution action based on the provided parameters.
        """

        try:

            iteration = kwargs.get("iteration", 0)

            # Sample parent and inspirations from database (Ray actor)
            parent, inspirations = ray.get(self.database.sample.remote())

            # Get artifacts for the parent program if available
            parent_artifacts = ray.get(self.database.get_artifacts.remote(parent.id))

            # Get island-specific top programs for prompt context (maintain island isolation)
            current_island = ray.get(self.database.get_current_island.remote())

            parent_island = parent.metadata.get("island", current_island)

            island_top_programs = ray.get(self.database.get_top_programs.remote(self.island_top_programs_limit, parent_island))
            island_previous_programs = ray.get(
                self.database.get_top_programs.remote(self.island_previous_programs_limit, parent_island)
            )

            # Build prompt
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in island_previous_programs],
                top_programs=[p.to_dict() for p in island_top_programs],
                inspirations=[p.to_dict() for p in inspirations],
                language=self.language,
                evolution_round=iteration,
                diff_based_evolution=self.diff_based_evolution,
                program_artifacts=parent_artifacts if parent_artifacts else None,
            )

            iteration_start = time.time()

            # Generate code modification
            llm_response = await self.llm_actor_client.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )

            logger.debug(f"LLM response for iteration {iteration}: {llm_response}")

            # TODO : we are using llm ensemble, so we need to handle the response accordingly
            llm_response = llm_response[0]

            # Parse the response
            if self.diff_based_evolution:
                diff_blocks = extract_diffs(llm_response)

                if not diff_blocks:
                    logger.warning(
                        f"Iteration {iteration+1}: No valid diffs found in response"
                    )
                    return None

                # Apply the diffs
                evovled_child_code = apply_diff(parent.code, llm_response)
                changes_summary = format_diff_summary(diff_blocks)
            else:
                # Parse full rewrite
                new_code = parse_full_rewrite(llm_response, self.language)

                if not new_code:
                    logger.warning(f"Iteration {iteration}: No valid code found in response")
                    return None

                # add # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END comments to enclose the code block if # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END not present in the new code 
                new_code = self._enclose_code_block(new_code)

                evovled_child_code = new_code
                changes_summary = "Full rewrite"

            # Check code length
            if len(evovled_child_code) > self.max_code_length:
                logger.warning(
                    f"Iteration {iteration}: Generated code exceeds maximum length "
                    f"({len(evovled_child_code)} > {self.max_code_length})"
                )
                return None

            child_id = str(uuid.uuid4())

            # Evaluate the child code
            exe_evaluation_result = await self.exe_critic.evaluate(
                evolved_program_code=evovled_child_code, program_id=child_id
            )
            llm_evaluation_result = None
            if self.use_llm_critic:
                llm_evaluation_result = await self.llm_critic.evaluate(
                    evolved_program_code=evovled_child_code, program_id=child_id
                )

                for name, value in llm_evaluation_result.metrics.items():
                    exe_evaluation_result.metrics[f"llm_{name}"] = value * self.llm_feedback_weight

            if self.artifacts_enabled and exe_evaluation_result.has_artifacts():
                self.artifacts.update(exe_evaluation_result.artifacts)
                logger.debug(f"Artifacts from execution critic: {exe_evaluation_result.artifacts}")

            if llm_evaluation_result and llm_evaluation_result.has_artifacts():
                self.artifacts.update(llm_evaluation_result.artifacts)
                logger.debug(f"Artifacts from LLM critic: {llm_evaluation_result.artifacts}")

            return ActionResult(
                child_program_dict={
                    "id": child_id,
                    "code": evovled_child_code,
                    "language": self.language,
                    "parent_id": parent.id,
                    "generation": parent.generation + 1,
                    "metrics": exe_evaluation_result.metrics,
                    "iteration_found": iteration,
                    "metadata": {
                        "island": parent_island,
                        "changes_summary": changes_summary,
                        "diff_based_evolution": self.diff_based_evolution,
                    },
                },
                parent_id=parent.id,
                iteration_time=time.time() - iteration_start,
                prompt=prompt,
                llm_response=llm_response,
                artifacts=self.artifacts if self.artifacts_enabled else None,
                iteration=iteration,
                error=None,
            )

        except Exception as e:
            logger.exception(f"Error in iteration {iteration}: {e}")
            return None
