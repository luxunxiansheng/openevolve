"""Simple template loader and LLM prompt builder.

- Loads templates from `opencontext/environment/templates/<name>.*` (by filename)
- Validates required placeholders using `render_template`/`placeholders` helpers
- Builds a system+user message pair enforcing the JSON contract

This is intentionally minimal and synchronous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .. import evolution_templates as et

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class TemplateNotFound(FileNotFoundError):
    pass


def load_template(name: str) -> str:
    """Load a template file by name (filename without extension).

    Looks for any file in the `templates` folder that starts with `name` and
    returns its text. Raises TemplateNotFound if not present.
    """
    candidates = list(TEMPLATES_DIR.glob(f"{name}.*"))
    if not candidates:
        raise TemplateNotFound(f"Template '{name}' not found in {TEMPLATES_DIR}")
    # prefer .md or .txt if multiple
    candidates.sort(key=lambda p: (p.suffix != ".md", p.suffix != ".txt", p.name))
    return candidates[0].read_text()


