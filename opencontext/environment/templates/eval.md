## Evaluation Template for Code

Please evaluate this {program} code and provide a single JSON response with numeric scores (0.0 to 1.0) for the criteria below:

Criteria
- correctness
- readability
- maintainability
- performance

Return only a JSON object with those keys (float values) and an optional short "notes" string for suggestions.

Code to evaluate:
```{program}
# Paste the code to evaluate here
```

Example response format:
```json
{
    "correctness": 0.0,
    "readability": 0.0,
    "maintainability": 0.0,
    "performance": 0.0,
    "notes": "Optional brief suggestions."
}
```