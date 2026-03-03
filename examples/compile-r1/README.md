# compile-r1

`compile-r1` is a Search-R1-style multi-turn tool-use example for code generation on HumanEval.

The model is trained to follow this protocol:

1. `<think>...</think>`
2. `<code>...</code>`
3. tool feedback is injected as `<result>...</result>`
4. `<answer>...</answer>` (final code)

## Files

- `generate_with_compile.py`: custom rollout + reward function
- `humaneval_format.py`: tag parsing / format checks
- `prepare_compile_r1_data.py`: convert `openai/openai_humaneval` to parquet
- `run_qwen2.5_3B.sh`: training launch script

## 1) Prepare data

```bash
cd /mnt/workspace/jkh/slime
python examples/compile-r1/prepare_compile_r1_data.py \
  --output-train examples/compile-r1/data/train.parquet \
  --output-test examples/compile-r1/data/test.parquet
```

Optional smoke dataset:

```bash
python examples/compile-r1/prepare_compile_r1_data.py \
  --output-train examples/compile-r1/data/train_smoke.parquet \
  --output-test examples/compile-r1/data/test_smoke.parquet \
  --max-rows 32
```

## 2) Run training

```bash
cd /mnt/workspace/jkh/slime
bash examples/compile-r1/run_qwen2.5_3B.sh
```

Default data paths are directory-based:

- train: `examples/compile-r1/data/train`
- test: `examples/compile-r1/data/test`

Or explicitly use the precheck config:

```bash
cd /mnt/workspace/jkh/slime
CONFIG_FILE=examples/compile-r1/configs/precheck.yaml \
  bash examples/compile-r1/run_qwen2.5_3B.sh
```

## Tool integration

Training integrates the local sandbox tool in `tools/python_sandbox/sandbox.py`.
By default this tool now uses `tools/RunPythonTool.py` (multi-process backend) at library level.

- During rollout, `<code>` content is extracted and executed in sandbox.
- Execution output is injected back as `<result>{...json...}</result>`.
- Reward re-runs `<answer>` code against HumanEval tests; pass => reward `1.0`.

The training script wires this through:

```bash
--custom-generate-function-path generate_with_compile.generate
--custom-rm-path generate_with_compile.reward_func
```

Backend-related env vars:

- `COMPILE_R1_SANDBOX_BACKEND` (`runpytool` / `subprocess`)
- `COMPILE_R1_RUNPY_MAX_WORKERS` (only used by `runpytool`)
