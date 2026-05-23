# Standalone Single-Pruning Scripts

These scripts are self-contained one-off pruning runners for local decoder-only Hugging Face checkpoints. Each script takes only:

```bash
python single_pruning/<script>.py /path/to/local_model /path/to/eval.json
```

Or edit `MODEL_PATH` and `EVAL_DATASET_PATH` near the top of the script and run it with no path arguments:

```bash
python single_pruning/<script>.py
```

They auto-launch with `torchrun` on up to 8 visible GPUs, evaluate the dense model with one full model copy per GPU and a sharded eval set, prune, save the pruned checkpoint immediately, reload that saved `pruned_model/` checkpoint from disk for evaluation, and print dense accuracy, pruned accuracy, whole-model sparsity, selected-linear sparsity, and output paths at the end. Increase `BATCH_SIZE` near the top of a script if GPU utilization is low and memory allows it.

## Scripts

```bash
python single_pruning/magnitude_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/gradient_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/wanda_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/prune_2of4_8gpu_exact.py /path/to/local_model /path/to/eval.json
```

Use the YAML-driven benchmark suite when you need all methods in one comparable benchmark table. Use these scripts for quick single-method pruning checks.
