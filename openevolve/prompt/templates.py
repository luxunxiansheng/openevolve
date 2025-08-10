"""
Prompt templates for OpenEvolve
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional


class TemplateKey(Enum):
    """Enum for template keys"""
    BASE_SYSTEM_MESSAGE = "base_system_message"
    ACTOR_SYSTEM_MESSAGE = "actor_system_message"
    CRITIC_SYSTEM_MESSAGE = "critic_system_message"
    DIFF_USER = "diff_user"
    FULL_REWRITE_USER = "full_rewrite_user"
    EVOLUTION_HISTORY = "evolution_history"
    PREVIOUS_ATTEMPT = "previous_attempt"
    TOP_PROGRAM = "top_program"
    INSPIRATIONS_SECTION = "inspirations_section"
    INSPIRATION_PROGRAM = "inspiration_program"
    EVALUATION_TEMPLATE = "evaluation_template"


# Base system message for evolution
BASE_SYSTEM_MESSAGE = "You are a helpful assistant designed to assist with code evolution tasks."

# Base system message template for actor
# This template is used to guide the LLM in generating code modifications
BASE_ACTOR_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""


# Base system message template for crtic
# This template is used to guide the LLM in evaluating code quality
BASE_CRITIC_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for formatting inspirations section
INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

# Template for formatting an individual inspiration program
INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```
Unique approach: {unique_features}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    TemplateKey.BASE_SYSTEM_MESSAGE.value: BASE_SYSTEM_MESSAGE,
    TemplateKey.ACTOR_SYSTEM_MESSAGE.value: BASE_ACTOR_SYSTEM_TEMPLATE,
    TemplateKey.CRITIC_SYSTEM_MESSAGE.value: BASE_CRITIC_SYSTEM_TEMPLATE,
    TemplateKey.DIFF_USER.value: DIFF_USER_TEMPLATE,
    TemplateKey.FULL_REWRITE_USER.value: FULL_REWRITE_USER_TEMPLATE,
    TemplateKey.EVOLUTION_HISTORY.value: EVOLUTION_HISTORY_TEMPLATE,
    TemplateKey.PREVIOUS_ATTEMPT.value: PREVIOUS_ATTEMPT_TEMPLATE,
    TemplateKey.TOP_PROGRAM.value: TOP_PROGRAM_TEMPLATE,
    TemplateKey.INSPIRATIONS_SECTION.value: INSPIRATIONS_SECTION_TEMPLATE,
    TemplateKey.INSPIRATION_PROGRAM.value: INSPIRATION_PROGRAM_TEMPLATE,
    TemplateKey.EVALUATION_TEMPLATE.value: EVALUATION_TEMPLATE,
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_key: TemplateKey) -> str:
        """Get a template by key"""
        if template_key.value not in self.templates:
            available_keys = [key.name for key in TemplateKey]
            raise ValueError(
                f"Template '{template_key.name}' not found. Available templates: {available_keys}"
            )
        return self.templates[template_key.value]

    def add_template(self, template_key: TemplateKey, template: str) -> None:
        """Add or update a template"""
        self.templates[template_key.value] = template

    def list_templates(self) -> List[str]:
        """List all available template keys"""
        return list(self.templates.keys())

    @staticmethod
    def get_all_template_keys() -> List[TemplateKey]:
        """Get all available template keys as enums"""
        return list(TemplateKey)
