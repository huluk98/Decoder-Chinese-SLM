# Git consolidation state before

## pwd
Decoder Only

## git status --short --branch
## codex/sft-pruning-on-main...origin/codex/sft-pruning-on-main
?? .vscode/
?? data/cleaned/
?? git_consolidation_state_before.md

## git remote -v
origin	https://github.com/huluk98/Decoder-Chinese-SLM.git (fetch)
origin	https://github.com/huluk98/Decoder-Chinese-SLM.git (push)

## git fetch --all --prune

## git branch -a
* codex/sft-pruning-on-main
  main
  remotes/origin/HEAD -> origin/main
  remotes/origin/codex/sft-pruning-on-main
  remotes/origin/codex/sft-pruning-workflows
  remotes/origin/main

## git branch -vv
* codex/sft-pruning-on-main 8fe91e4 [origin/codex/sft-pruning-on-main] Align pruning benchmark with CMC protocol
  main                      f206d4e Add SFT and pruning workflows

## git log --oneline --decorate --graph -n 30
* 8fe91e4 (HEAD -> codex/sft-pruning-on-main, origin/codex/sft-pruning-on-main) Align pruning benchmark with CMC protocol
* 4a679bc Clarify pruning active parameter reporting
* 1acdc14 Batch prompt response benchmark generation
* cce0f73 Fix pruning benchmark workflow
* 1349bc3 Add sequential pruning benchmark wrapper
* fb93d85 Add TensorRT edge deployment pipeline
* 766e53c Add Qwen2.5-Instruct SFT pipeline
* 689ace1 Harden SFT evaluation audit pipeline
* e036e5f Report active parameters in pruning benchmark
* af17483 Add pruning benchmark suite
* 30b52c4 Accept anchor field in SFT datasets
* 8130702 Add 8 GPU contrastive SFT launcher
* 7022075 Add repeated SFT benchmark eval
* 8f4c310 Resolve SFT final eval checkpoint path
* 3798c80 Run SFT eval only after training
* 6e7af74 Add project about page
* fd07158 Canonicalize command eval comparisons
* b8d953b Shard prompt response eval across GPUs
* 245e7a8 Make prompt response exact match default
* ef16c9c Add exact match prompt response eval
* 848d988 Optimize SFT trainer for 8 GPU command tuning
* 3525a1c Split prompt response eval from C-Eval
* d061ef6 Normalize SFT records for model format
* 621c03c Support local files in C-Eval script
* 046a079 Add epoch control for SFT
* 3274cd0 Support JSON files for regular SFT
* f484b95 Add references and citation metadata
* 24775bc Add local generation dataset overrides
* 13ed2ff Split C-Eval results by category
* c2a89cb Add latest C-Eval results
