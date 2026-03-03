# Python Sandbox Tool

This tool executes candidate Python code against HumanEval-style tests with resource limits.

## Location

- Python module: `tools/python_sandbox/sandbox.py`
- Import API: `from tools.python_sandbox import run_humaneval_in_sandbox`
- Multi-process backend module: `tools/RunPythonTool.py`

## Input (Python API)

`run_humaneval_in_sandbox(...)` accepts:

- `code` (`str`): candidate code from model output (`<code>` or `<answer>` body)
- `prompt` (`str`): HumanEval task prompt (function signature + docstring)
- `test` (`str`): HumanEval test program containing `check(candidate)`
- `entry_point` (`str`): target function name
- `timeout_seconds` (`int`, optional, default `6`)
- `memory_mb` (`int`, optional, default `768`)
- `max_output_chars` (`int`, optional, default `8000`)

## Output (Python API)

Returns a `dict`:

- `passed` (`bool`): whether all checks pass
- `status` (`str`): `passed` / `runtime_error` / `timeout` / `safety_reject` / `sandbox_error`
- `error` (`str`): concise error message
- `stdout` (`str`): captured stdout (truncated if too long)
- `stderr` (`str`): captured stderr (truncated if too long)
- `execution_seconds` (`float`): wall-clock runtime
- `returncode` (`int`): sandbox process return code
- optional fields: `traceback`, `used_prompt_prefix`

## CLI Usage

### 1) JSON payload mode

```bash
python tools/python_sandbox/sandbox.py \
  --input-json /path/to/payload.json
```

`payload.json` format:

```json
{
  "code": "def add(a, b):\n    return a + b",
  "prompt": "def add(a, b):\n    \"\"\"return sum\"\"\"\n",
  "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
  "entry_point": "add"
}
```

### 2) File/inline mode

```bash
python tools/python_sandbox/sandbox.py \
  --code-file /tmp/candidate.py \
  --prompt-file /tmp/prompt.py \
  --test-file /tmp/test.py \
  --entry-point add
```

CLI prints one JSON object (same schema as API output).

## Backend Selection

Default backend is `runpytool` (library adapter over `tools/RunPythonTool.py`).

- `SLIME_PY_SANDBOX_BACKEND=runpytool` (default)
- `SLIME_PY_SANDBOX_BACKEND=subprocess` (fallback legacy runner)
- `RUNPY_TOOL_MAX_WORKERS=4` (effective only in `runpytool` backend)
