"""
Prompt templates for OpenEvolve - Clean and organized implementation
"""

import os
from pathlib import Path
from typing import Optional


class Templates:
    """Template constants with clear organization and type safety"""

    # System message templates
    BASE_SYSTEM = "base_system_message"
    ACTOR_SYSTEM = "actor_system_message"
    CRITIC_SYSTEM = "critic_system_message"

    # User message templates
    DIFF_USER = "diff_user"
    FULL_REWRITE_USER = "full_rewrite_user"

    # Evolution history templates
    EVOLUTION_HISTORY = "evolution_history"
    PREVIOUS_ATTEMPT = "previous_attempt"
    TOP_PROGRAM = "top_program"
    INSPIRATIONS_SECTION = "inspirations_section"
    INSPIRATION_PROGRAM = "inspiration_program"

    # Evaluation templates
    EVALUATION = "evaluation"


# System Messages
_BASE_SYSTEM_MESSAGE = "You are a helpful assistant designed to assist with code evolution tasks."

_ACTOR_SYSTEM_MESSAGE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics."""

_CRITIC_SYSTEM_MESSAGE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User Templates
_DIFF_USER_TEMPLATE = """# Current Program Information
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

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements."""

_FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
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
```"""

# Evolution History Templates
_EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}"""

_PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}"""

_TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}"""

_INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}"""

_INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```
Unique approach: {unique_features}"""

# Evaluation Templates
_EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
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
}}"""

# Clean template registry
TEMPLATES = {
    # System templates
    Templates.BASE_SYSTEM: _BASE_SYSTEM_MESSAGE,
    Templates.ACTOR_SYSTEM: _ACTOR_SYSTEM_MESSAGE,
    Templates.CRITIC_SYSTEM: _CRITIC_SYSTEM_MESSAGE,
    # User templates
    Templates.DIFF_USER: _DIFF_USER_TEMPLATE,
    Templates.FULL_REWRITE_USER: _FULL_REWRITE_USER_TEMPLATE,
    # Evolution templates
    Templates.EVOLUTION_HISTORY: _EVOLUTION_HISTORY_TEMPLATE,
    Templates.PREVIOUS_ATTEMPT: _PREVIOUS_ATTEMPT_TEMPLATE,
    Templates.TOP_PROGRAM: _TOP_PROGRAM_TEMPLATE,
    Templates.INSPIRATIONS_SECTION: _INSPIRATIONS_SECTION_TEMPLATE,
    Templates.INSPIRATION_PROGRAM: _INSPIRATION_PROGRAM_TEMPLATE,
    # Evaluation templates
    Templates.EVALUATION: _EVALUATION_TEMPLATE,
}


class TemplateManager:
    """Simple template manager using dictionary lookup"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = TEMPLATES.copy()

        # Load additional templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template

    def list_templates(self) -> list[str]:
        """List all available template names"""
        return list(self.templates.keys())
