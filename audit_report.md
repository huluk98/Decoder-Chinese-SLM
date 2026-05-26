# SFT And Exact-Match Evaluation Audit

## Purpose

This audit checks whether a jump from about 83% to about 96% exact-match accuracy is likely to be a real model improvement or a pipeline artifact. The task is smart-home command generation, where many natural-language prompts can map to the same normalized response, but the main metric should remain whitespace-insensitive exact match.

## Key Findings

1. `run_sft_8gpu.sh` launches `scripts/train.py`, but `scripts/train.py` detects SFT configs and dispatches into `scripts/sft.py`. This is functional, but it made the actual SFT path less obvious. The SFT run now writes `run_config.json` with the resolved base model, train/eval files, output directory, model class/type, parameter count, checkpoint timestamps, seeds, and training arguments.

2. SFT now saves exactly one final checkpoint at `output_dir/final/`, with tensors, tokenizer files, config, `trainer_state.pt`, and `checkpoint_manifest.json`. The launch scripts require this exact final checkpoint and no longer fall back to `latest` or `step-*`, which removes the stale SFT checkpoint failure mode.

3. Evaluation previously wrote into a fixed output directory such as `eval/final_prompt_response_benchmark`. That could overwrite previous prediction files and make yesterday/today comparisons ambiguous. Evaluation now writes each run into a unique timestamped child directory and writes `latest_eval_dir.txt` plus compatibility summaries in the parent directory.

4. The old default exact-match comparison mode was `command`, which canonicalized smart-home wording variants. That is useful for semantic command evaluation, but it is more permissive than whitespace-insensitive exact match and can inflate accuracy. The default is now `whitespace`. Use `--comparison-mode command` only when intentionally measuring semantic command equivalence.

5. Split leakage could not be checked unless the evaluator knew the train split. The 8-GPU SFT launchers now pass `--train-file` into eval when a separate eval file exists. The evaluator writes `split_audit.json` and fails by default if train/validation/test share exact prompts, exact prompt-response pairs, or anchor ids. Response overlap alone is logged but not treated as leakage because the task is many-to-one.

6. SFT data shuffling was not fully controlled by the config seed. `DistributedSampler` used its default seed unless explicitly set. The dataloader now receives a configured data seed, and SFT accepts `--seed` and `--data-seed`.

7. Evaluation now logs all generation settings: `do_sample`, `num_beams`, `temperature`, `top_p`, `top_k`, `max_new_tokens`, `repetition_penalty`, `eos_token_id`, and `pad_token_id`. Greedy decoding remains the deterministic default.

8. The denominator is kept as the full eval set size. Empty predictions, invalid structured outputs, and generation errors are counted in `metrics.json`; they are not silently excluded.

9. Qwen2.5-0.5B-Instruct now has a separate SFT/eval path. It uses `tokenizer.apply_chat_template(..., add_generation_prompt=True)` during both training and evaluation, saves only `output_dir/final/`, and rejects the legacy `<|user|>` / `<|assistant|>` / `<|system|>` / `<|eos|>` formatting tokens. Previous Qwen-Instruct scores from the legacy formatter should be treated as invalid until rerun through `scripts/sft_qwen25_instruct.py` and `scripts/eval_qwen25_instruct.py`.

10. Qwen2.5-0.5B-Instruct pruning now has a separate benchmark path. `scripts/prune_qwen25_instruct.py` uses Qwen chat-template calibration data for Wanda/gradient pruning, and prune+retune calls `scripts/sft_qwen25_instruct.py --pruning-mask` so fixed masks are preserved during Qwen SFT. The default pruning configs now target full-model 50% sparsity with no protected floating-point parameters, and the benchmark runners validate the configured sparsity denominator plus zero masked-weight violations before each phase.

## New Diagnostic Outputs

Each eval run writes:

- `run_config.json`: model/checkpoint/tokenizer identity, seeds, generation settings, git commit, checkpoint file timestamps.
- `split_audit.json`: split sizes, unique prompt/response counts, duplicate counts, and leakage overlaps.
- `metrics.json`: exact-match accuracy, correct/incorrect counts, empty predictions, invalid outputs, generation errors, average generated length, average label length.
- `prediction_debug.csv`: `id`, `prompt`, `raw_prediction`, `normalized_prediction`, `raw_label`, `normalized_label`, `exact_match`, `generated_length`, `label_length`.
- `prompt_response_eval_predictions.jsonl`: full per-example records.

For repeated benchmarks, `run_01`, `run_02`, etc. keep separate predictions, while the top-level eval folder also keeps the latest run's debug CSV and the aggregate metrics.

The Qwen2.5-Instruct evaluator writes the Qwen-specific equivalents:

- `qwen25_instruct_eval_summary.json`
- `qwen25_instruct_predictions.jsonl`
- `qwen25_instruct_prediction_debug.csv`
- `failed_examples_20.json`

## How To Compare 83% vs 96%

Run:

```bash
python scripts/compare_predictions.py \
  /path/to/yesterday/prompt_response_eval_predictions.jsonl \
  /path/to/today/prompt_response_eval_predictions.jsonl \
  --output-json runs/eval/compare_83_vs_96.json
```

Interpretation:

- Many identical predictions with different model checkpoints is suspicious.
- A large number of wrong-to-correct changes with different raw predictions can be legitimate.
- Identical raw predictions but different accuracy usually points to changed normalization, changed labels, or a comparison bug.
- Different eval files or split audit fingerprints mean the 83% and 96% runs are not directly comparable.

## Current Recommendation

Treat the 96% as unconfirmed until the new audit outputs show:

1. The eval checkpoint is the intended trained checkpoint.
2. The eval file is unchanged.
3. No train/test prompt, prompt-response, or anchor leakage exists.
4. Accuracy remains high under `--comparison-mode whitespace`.
5. Prediction comparison shows real wrong-to-correct changes rather than cache/checkpoint reuse.
