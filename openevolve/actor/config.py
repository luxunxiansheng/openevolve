from dataclasses import dataclass


@dataclass
class EvolutionActorConfig:
    language: str = "python"
    iteration: int = 100
    diff_based_evolution: bool = False
    max_code_length: int = 20480
    use_llm_critic: bool = True
    llm_feedback_weight: float = 0.1
    artifacts_enabled: bool = True
    island_top_programs_limit: int = 3
    island_diverse_programs_limit: int = 3
    
    
