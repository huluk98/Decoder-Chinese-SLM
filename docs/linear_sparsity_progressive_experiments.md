# Linear Sparsity and Progressive Recovery Experiments

## Purpose

This experiment block evaluates whether SCENIC IoT command-normalization accuracy is stable across multiple Linear-weight sparsity levels. It adds a dense baseline, one-shot magnitude pruning, and staged progressive magnitude pruning at 30% and 50% target sparsity, with EM@1 and EM@5 reported overall and by command difficulty.

The outputs are designed for manuscript revision tables: per-example predictions, summary metrics with bootstrap confidence intervals, a paper-ready pivot table, progressive stage logs, saved masks/checkpoints, and plots.

## Experimental Conditions

The runner supports the model family names:

- `decoder_only`
- `encoder_decoder`
- `encoder_only`

The core conditions are:

- `dense` at `0.0` sparsity: no pruning, benchmark evaluation only.
- `oneshot` at `0.3` and `0.5`: magnitude pruning is applied once before benchmark evaluation.
- `progressive` at `0.3` and `0.5`: masks are updated stage by stage. If `--recovery_train_path` is supplied, the model performs recovery fine-tuning after each stage and after the final stage. If no recovery data is supplied, the runner still applies staged masks and writes progressive logs with empty loss/validation fields.

## Sparsity Levels

The levels `0%`, `30%`, and `50%` give a dense reference point, a moderate pruning condition, and the paper's original 50% stress condition. This makes it possible to report whether accuracy degradation is monotonic and whether the original one-shot 50% conclusion is robust.

## Pruning Scope

The new experiment prunes only `torch.nn.Linear.weight` tensors selected by the repository pruning protocol.

Excluded by default:

- bias terms
- embeddings and positional embeddings
- LayerNorm/RMSNorm parameters
- `classifier` / `classification_head`
- `lm_head` and `output_head`
- final response projection style heads
- other protected modules already excluded by the repository pruning utilities

`--prune_output_heads` is false by default. Use it only for an explicit ablation where output heads are allowed into the pruning mask.

The summary reports both:

- `targeted_linear_sparsity_actual`: zero fraction inside selected Linear weights.
- `whole_model_sparsity_actual`: zero fraction over all model parameters.

These values differ because embeddings, norms, biases, and heads are excluded from the target mask.

## Pruning Methods

This block uses magnitude pruning:

- Per-layer unstructured pruning by default.
- Each selected Linear layer reaches the requested sparsity.
- `--global_pruning` switches to one global threshold over all selected Linear weights.

Masks are saved as `.pt` files and reapplied after every optimizer step during progressive recovery fine-tuning, so pruned weights remain zero. Regrowth is disabled unless `--regrowth` is passed.

Progressive schedules:

- target `0.30`: `0.10 -> 0.20 -> 0.30`
- target `0.50`: `0.10 -> 0.20 -> 0.30 -> 0.40 -> 0.50`

## EM@1 and EM@5

EM@1 is true when the normalized top-1 prediction exactly equals the normalized target.

EM@5 is true when the normalized target appears in the normalized top-5 candidate list.

For decoder-only and encoder-decoder models, candidates are generated with beam search. Use at least:

```bash
--num_beams 5 --num_return_sequences 5
```

For encoder-only models, the runner uses the top five class labels from `model.config.id2label` as canonical response candidates.

`--normalization_mode command` uses the repository command canonicalizer. Other supported modes are `exact`, `whitespace`, `normalized`, and `punctuation`.

## Difficulty Labels

The benchmark may contain one of these columns/fields:

- `difficulty`
- `complexity`
- `level`

Labels are normalized to lowercase and must be `easy`, `medium`, or `hard`.

If the benchmark does not include labels, pass:

```bash
--benchmark_difficulty_path path/to/difficulty.csv
```

The difficulty file may be CSV, JSON, or JSONL and should contain:

- `id,difficulty`
- `sample_id,difficulty`
- or `input,difficulty`

The evaluator joins by sample id first, then exact input string. It raises an error if a label cannot be joined.

To create a manual labeling template:

```bash
python scripts/create_benchmark_difficulty_template.py \
  --benchmark_path data/benchmarks/iot_instruction_benchmark_200.json \
  --output_dir data/benchmarks
```

## Reproducing Runs

For the revision run from one base model path, use the one-line wrapper:

```bash
PYTHON=/path/to/env/bin/python bash run_linear_sparsity_revision_from_base.sh /path/to/base_model
```

This wrapper keeps the original four methods (`magnitude`, `wanda`, `taylor`, `2of4`) as one-shot-only runs and now sweeps the native-method pruning targets with `SPARSITY_LEVELS="0.3 0.5"` by default. Exact NVIDIA `2of4` is still a fixed 50% structured condition; a requested 30% native sweep records the request but reports the achieved 50% 2:4 sparsity rather than treating it as a true 30% unstructured result. It then runs the added Linear sparsity experiment with one recovery epoch per progressive stage and one final recovery epoch.

Example:

```bash
python scripts/run_sparsity_experiments.py \
  --experiment_name scenic_linear_sparsity_0_30_50 \
  --model_family encoder_decoder \
  --model_checkpoint PATH_TO_CHECKPOINT \
  --benchmark_path PATH_TO_BENCHMARK \
  --benchmark_difficulty_path PATH_TO_DIFFICULTY_LABELS \
  --sparsity_levels 0 0.3 0.5 \
  --pruning_modes dense oneshot progressive \
  --prune_scope linear_weights \
  --prune_method magnitude \
  --recovery_epochs_per_stage 1 \
  --final_recovery_epochs 2 \
  --num_beams 5 \
  --num_return_sequences 5 \
  --seed 42 \
  --output_dir results/scenic_linear_sparsity_0_30_50
```

For actual progressive recovery fine-tuning, add:

```bash
--recovery_train_path data/scenic/SCENIC_full_training_dataset.json \
--validation_path path/to/validation.json
```

## Outputs

The runner writes:

- `predictions_{model_family}_{pruning_mode}_{sparsity}_{seed}.csv`
- `summary_metrics.csv`
- `paper_table_sparsity_difficulty.csv`
- `progressive_logs_{model_family}_{target_sparsity}_{seed}.csv`
- `checkpoints/`
- `masks/`
- `figures/em1_vs_sparsity.png`
- `figures/em5_vs_sparsity.png`
- `figures/difficulty_em1_bar.png`
- `figures/difficulty_em5_bar.png`

You can regenerate plots from an existing summary:

```bash
python scripts/plot_sparsity_results.py \
  --summary_metrics results/scenic_linear_sparsity_0_30_50/summary_metrics.csv
```

## Citing Results in the Paper

Use `paper_table_sparsity_difficulty.csv` for the main revision table. Report dense, one-shot 30%, one-shot 50%, progressive 30%, and progressive 50% rows for each model family.

When discussing retention, cite the `*_retention_*` columns in `summary_metrics.csv`. These values are computed as pruned EM divided by dense EM for the same model family and difficulty group.

For confidence intervals, cite the overall CI columns directly. Difficulty-level CIs are written when the group count is at least 20; smaller groups keep the score but mark the CI as `insufficient_n`.
