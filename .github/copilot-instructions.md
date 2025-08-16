# GitHub Copilot Instructions

## Code Organization Philosophy

### Modular Package Structure
- Break large files into focused, single-responsibility modules
- Each file should be short and readable (ideally under 200 lines)
- Group related functionality into packages with clear `__init__.py` exports
- Prefer composition over inheritance

### File Structure Guidelines
- **One class per file** when possible
- **Clear naming** that reflects the module's purpose
- **Focused responsibility** - each module should do one thing well
- **Clean imports** - use package-level imports through `__init__.py`

### Code Style Preferences
- Use type hints consistently
- Prefer dataclasses for simple data structures
- Keep methods short and focused
- Use descriptive variable and function names
- Add docstrings for all public classes and methods

### Example Structure
```
package/
├── __init__.py          # Public API exports
├── core_module.py       # Main functionality
├── data_models.py       # Data structures
├── utilities.py         # Helper functions
└── exceptions.py        # Custom exceptions
```

### When Refactoring
1. Identify logical boundaries between responsibilities
2. Extract classes/functions into separate modules
3. Create a package with clear public API
4. Maintain backward compatibility through imports
5. Keep related code physically close

### Avoid
- Monolithic files with multiple unrelated classes
- Deep inheritance hierarchies
- Circular imports
- Hard-coded values (make them configurable)
- Mixing different levels of abstraction in one file

## Implementation Examples
- See `opencontext/environment/program_evolution/` package for reference
- Each module has a single, clear purpose
- Package provides clean public API
- Easy to navigate and understand
