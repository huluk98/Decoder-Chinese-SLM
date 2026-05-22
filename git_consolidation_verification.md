# Git consolidation verification

## git fetch --all --prune

## Commits from sft-pruning-on-main not in consolidation

## Commits from sft-pruning-workflows not in consolidation

## Diff from sft-pruning-on-main to consolidation
 data/cleaned/619_Luke_fix_report.json  |   901 ++
 data/cleaned/619_Luke_fixed_all.json   | 18062 +++++++++++++++++++++++++++++++
 data/cleaned/619_Luke_fixed_dedup.json | 13898 ++++++++++++++++++++++++
 3 files changed, 32861 insertions(+)

## Diff from sft-pruning-workflows to consolidation
 .gitignore                                         |     2 +-
 ABOUT.md                                           |    60 +
 CITATION.cff                                       |    27 +
 README.md                                          |   483 +-
 README_TRT_EDGE.md                                 |   153 +
 audit_report.md                                    |    74 +
 configs/contrastive_sft.yaml                       |     3 +-
 configs/contrastive_sft_8gpu.yaml                  |    46 +
 configs/prune_qwen25_50.yaml                       |    23 +
 configs/pruning_benchmark.yaml                     |    42 +
 configs/qwen25_instruct_pruning_benchmark.yaml     |    44 +
 configs/sft.yaml                                   |     3 +-
 configs/sft_0p2b_8gpu.yaml                         |    44 +
 configs/sft_qwen25_0p5b_instruct.yaml              |    42 +
 data/cleaned/619_Luke_fix_report.json              |   901 +
 data/cleaned/619_Luke_fixed_all.json               | 18062 +++++++++++++++++++
 data/cleaned/619_Luke_fixed_dedup.json             | 13898 ++++++++++++++
 data/sample_zh_dialog.jsonl                        |     1 -
 eval_results/ceval/latest/README.md                |    40 +
 .../ceval/latest/ceval_category_summary.csv        |     5 +
 eval_results/ceval/latest/ceval_predictions.csv    |  1347 ++
 eval_results/ceval/latest/ceval_summary.json       |   461 +
 requirements_edge.txt                              |    33 +
 run_contrastive_sft_8gpu.sh                        |   115 +
 run_pruning_benchmark_suite.sh                     |    14 +
 run_qwen25_instruct_pruning_benchmark_8gpu.sh      |    30 +
 run_qwen25_instruct_sft_8gpu.sh                    |   116 +
 run_sft_8gpu.sh                                    |   106 +
 run_sft_debug_1gpu.sh                              |    12 +
 scripts/benchmark_trt_decoder.py                   |   286 +
 scripts/build_trt_engines.py                       |   543 +
 scripts/compare_predictions.py                     |   132 +
 scripts/compare_pytorch_onnx_trt.py                |   321 +
 scripts/eval_ceval.py                              |   133 +-
 scripts/eval_prompt_response.py                    |  1331 ++
 scripts/eval_qwen25_instruct.py                    |  1001 +
 scripts/export_decoder_onnx.py                     |   364 +
 scripts/generate.py                                |   176 +-
 scripts/inspect_pruning_report.py                  |    76 +
 scripts/prune.py                                   |    98 +-
 scripts/prune_qwen25_instruct.py                   |   336 +
 scripts/run_all_trt_pipeline.sh                    |   105 +
 scripts/run_pruning_benchmark.py                   |   892 +
 scripts/run_pruning_benchmark_8way.sh              |    88 +
 scripts/run_qwen25_instruct_pruning_benchmark.py   |   934 +
 scripts/sft.py                                     |  1055 +-
 scripts/sft_qwen25_instruct.py                     |   691 +
 scripts/train.py                                   |    27 +-
 scripts/trt_edge_common.py                         |   455 +
 src/chatlm_decoder/__init__.py                     |     1 -
 src/chatlm_decoder/command_eval.py                 |   238 +
 src/chatlm_decoder/pruning.py                      |   503 +-
 src/chatlm_decoder/qwen25_instruct_data.py         |   315 +
 src/chatlm_decoder/sft_data.py                     |   225 +-
 54 files changed, 46355 insertions(+), 158 deletions(-)

## Files present on sft-pruning-workflows but missing from consolidation

None.

## Conflict resolution notes

- `codex/sft-pruning-on-main` merge conflicts were resolved in favor of the CMC-complete pruning/eval workflow where it overlapped with older `origin/main` Qwen pruning files.
- `README.md` and `audit_report.md` kept the newer dense-baseline, CMC-comparability, 50% prunable-linear sparsity, and checkpoint-validation documentation.
- `configs/qwen25_instruct_pruning_benchmark.yaml`, `run_qwen25_instruct_pruning_benchmark_8gpu.sh`, `scripts/prune_qwen25_instruct.py`, `scripts/run_pruning_benchmark_8way.sh`, `scripts/run_qwen25_instruct_pruning_benchmark.py`, and `scripts/sft_qwen25_instruct.py` kept the validator-heavy CMC-compatible versions from `codex/sft-pruning-on-main`.
- `codex/sft-pruning-workflows` had unrelated history and no file paths missing from the consolidation tree. It was merged with the `ours` strategy to record its commits as included while preserving the newer `origin/main` plus CMC pruning code.

## Basic checks

- `python3 -m compileall .` passed.
- `bash -n run_pruning_benchmark_suite.sh` passed.
- `bash -n run_qwen25_instruct_pruning_benchmark_8gpu.sh` passed.
- `bash -n scripts/run_pruning_benchmark_8way.sh` passed.
- Repository discovery found `pyproject.toml`; no `tests/`, `pytest.ini`, `tox.ini`, `Makefile`, `package.json`, or `.github/workflows/*` files were present to run additional local test/lint commands.
