# Standalone Single-Pruning Scripts

These scripts are self-contained one-off pruning runners for local decoder-only Hugging Face checkpoints. Each script takes only:

```bash
python single_pruning/<script>.py /path/to/local_model /path/to/eval.json
```

They auto-launch with `torchrun` on up to 8 visible GPUs, evaluate the dense model, prune, evaluate the pruned model, and save the pruned checkpoint plus reports next to the model path.

## Scripts

```bash
python single_pruning/magnitude_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/gradient_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/wanda_prune_8gpu_exact.py /path/to/local_model /path/to/eval.json
python single_pruning/prune_2of4_8gpu_exact.py /path/to/local_model /path/to/eval.json
```

Use the YAML-driven benchmark suite when you need all methods in one comparable benchmark table. Use these scripts for quick single-method pruning checks.
