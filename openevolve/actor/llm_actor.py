import logging
import time
import uuid
from openevolve.actor.actor import Actor, Result
from openevolve.database import Program, ProgramDatabase
from openevolve.llm.llm_interface import LLMInterface
from openevolve.prompt.sampler import PromptSampler

from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class LLMActor(Actor):
    def __init__(self,
                 database: ProgramDatabase,  
                 prompt_sampler: PromptSampler,
                 llm_client: LLMInterface,
                 language: str="python",
                max_code_length: int= 2048,
    ):
        self.database = database
        self.prompt_sampler = prompt_sampler
        self.llm_client = llm_client
        self.language = language # Default language, can be overridden
        self.max_code_length = max_code_length


    async def act(self, **kwargs) -> Result:
        """
        Perform the LLM action based on the provided parameters.
        
        This method samples a parent program and inspirations, builds a prompt,
        and generates a new program using the LLM client.
        """
        try:
            
            # Sample parent and inspirations from database
            parent, inspirations = self.database.sample()

            # Get artifacts for the parent program if available
            parent_artifacts = self.database.get_artifacts(parent.id)

                       # Get island-specific top programs for prompt context (maintain island isolation)
            parent_island = parent.metadata.get("island", self.database.current_island)
            island_top_programs = self.database.get_top_programs(5, island_idx=parent_island)
            island_previous_programs = self.database.get_top_programs(3, island_idx=parent_island)

            evaluation_round = kwargs.get("iteration", 0)
            diff_based_evolution = kwargs.get("diff_based_evolution", False)
            iteration = kwargs.get("iteration", 0)


            # Build prompt
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in island_previous_programs],
                top_programs=[p.to_dict() for p in island_top_programs],
                inspirations=[p.to_dict() for p in inspirations],
                language=self.language,
                evolution_round=evaluation_round,
                diff_based_evolution=diff_based_evolution,
                program_artifacts=parent_artifacts if parent_artifacts else None,
            )

            # Generate new program using LLM client
            llm_response = await self.llm_client.generate_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )


            if diff_based_evolution:
                diff_blocks = extract_diffs(llm_response)

                if not diff_blocks:
                    logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                    return None

                # Apply the diffs
                child_code = apply_diff(parent.code, llm_response)
                changes_summary = format_diff_summary(diff_blocks)
            else:
                # Parse full rewrite
                new_code = parse_full_rewrite(llm_response, self.language)

                if not new_code:
                    logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                    return None

                child_code = new_code
                changes_summary = "Full rewrite"

            # Check code length
            if len(child_code) > self.max_code_length:
                logger.warning(
                    f"Iteration {iteration+1}: Generated code exceeds maximum length "
                    f"({len(child_code)} > {self.max_code_length})"
                )
                return None
            

            




       

        except Exception as e:
            logger.error(f"Error in LLMActor: {e}")
            return Result(success=False, error=str(e))

