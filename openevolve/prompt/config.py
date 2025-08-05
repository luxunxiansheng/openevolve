
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = 500  # Suggest simplifying if program exceeds this many characters
    include_changes_under_chars: Optional[int] = 100  # Include change descriptions in features if under this length
    concise_implementation_max_lines: Optional[int] = 10  # Label as "concise" if program has this many lines or fewer
    comprehensive_implementation_min_lines: Optional[int] = 50  # Label as "comprehensive" if program has this many lines or more
    
    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = None  # Deprecated: use suggest_simplification_after_chars
