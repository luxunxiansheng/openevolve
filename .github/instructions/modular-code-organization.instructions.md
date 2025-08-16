# Modular Code Organization Instructions

## Core Principles

When writing or refactoring Python code, always follow these principles:

### 1. Single Responsibility Modules
- Each file should have one clear purpose
- Keep files under 200 lines when possible
- If a file grows large, break it into a package with focused modules

### 2. Package Structure
```
package_name/
├── __init__.py          # Clean public API
├── core.py             # Main functionality
├── models.py           # Data structures
├── utils.py            # Helper functions
└── exceptions.py       # Custom exceptions
```

### 3. Naming Conventions
- Module names should clearly indicate their purpose
- Use descriptive names: `program_extractor.py` not `extractor.py`
- Package names should be short but clear

### 4. Import Strategy
- Use package-level imports through `__init__.py`
- Keep imports at the package level clean and minimal
- Export only what's needed in the public API

### 5. Configuration Over Hard-coding
- Make display lengths, thresholds, and limits configurable
- Use dataclasses or config objects for parameters
- Avoid magic numbers in code

## Example Implementation

This modular style was successfully applied to `opencontext/environment/program_evolution/`:

**Before**: One 500+ line file with multiple classes
**After**: Clean package with focused modules:
- `program_extractor.py` - Extracts programs from LLM responses
- `template_manager.py` - Handles template loading
- `prompt_builder.py` - Builds prompts from templates
- `evolution_engine.py` - Main orchestrator
- `__init__.py` - Clean public API

## When to Apply
- Any file over 200 lines
- Files with multiple unrelated classes
- Code with mixed abstraction levels
- Hard-coded configuration values
- Complex inheritance hierarchies

## Benefits
✅ Easy to navigate and understand
✅ Better testing (each component testable)
✅ Improved maintainability
✅ Clear separation of concerns
✅ Reusable components
