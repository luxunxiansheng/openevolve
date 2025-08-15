# diff.md — Prompt template for suggesting targeted program improvements

**Placeholders used in this template**
- `metrics`: summary of current performance metrics
- `improvement_areas`: short list of areas needing improvement
- `artifacts`: optional additional artifacts (logs, outputs, images)
- `evolution_history`: prior attempts, top programs, inspirations
- `language`: programming language identifier for the code block
- `current_program`: the source code to analyze


## Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}


## Program Evolution History
{evolution_history}


## Current Program
```{language}
{current_program}
```


## Task
Suggest improvements to the program that will lead to better performance on the specified metrics.


## Output contract (REQUIRED)
You MUST return ONLY a single JSON array (no surrounding text). Each array element must be an object with these fields:

- `search` (string): an exact substring of `{current_program}` to replace.
- `replace` (string): the replacement text.
- `rationale` (string): one short sentence (<= 30 words) explaining the change.
- `order` (integer, optional): application order for multiple changes (starting at 1).

If there are no changes, return an empty JSON array: `[]`.

Example (the LLM must emit exactly this JSON — no surrounding commentary):

```json
[
  {
    "search": "for i in range(m):\\n\\tfor j in range(p):\\n\\t\\tfor k in range(n):\\n\\t\\t\\tC[i, j] += A[i, k] * B[k, j]",
    "replace": "for i in range(m):\\n\\tfor k in range(n):\\n\\t\\tfor j in range(p):\\n\\t\\t\\tC[i, j] += A[i, k] * B[k, j]",
    "rationale": "Reorders loops for better memory locality and cache behavior.",
    "order": 1
  }
]
```


### Additional rules (enforced by the caller)
- Each `search` string must match exactly one contiguous substring in `{current_program}`; if it does not match, the caller will reject the response.
- Make minimal, targeted changes. Do not rename variables or change program inputs/outputs unless explicitly requested.
- Keep `rationale` short and factual.
- If multiple edits are provided, `order` decides the application order. If omitted, apply in array order.

You can still include human-readable explanations in your internal reasoning, but the model's final reply MUST be the JSON array only (or `[]` if no changes). Failure to comply will cause automatic rejection and a retry.


You MUST use the exact SEARCH/REPLACE diff format shown below in your reasoning; the final emitted JSON should contain the exact `search` and `replace` texts that would appear in those blocks.

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format (for human reasoning):
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

**IMPORTANT:** Do not rewrite the entire program — focus on targeted improvements.
