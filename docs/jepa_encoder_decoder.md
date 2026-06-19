# Encoder-Decoder Text JEPA

This path adds a real JEPA-style objective for encoder-decoder models such as
T5 or mT5. It is separate from causal LM pretraining, SFT, and contrastive SFT.

## Core Idea

```text
context text -> trainable encoder -> context embedding
target text  -> EMA target encoder -> target embedding
context embedding + target query -> predictor -> predicted target embedding
loss = distance(predicted target embedding, stopgrad(target embedding))
```

The model does not reconstruct target tokens during JEPA pretraining. It learns
to predict the representation of the target text from the representation of the
context text.

## Why This Is Different From Current Contrastive SFT

Contrastive SFT in `scripts/sft.py` pulls anchor/positive embeddings together
and pushes negatives away while still optimizing response generation. JEPA makes
a conditional latent prediction: given context and a query for what target is
being predicted, the predictor regresses to the target encoder's latent state.

## Run

```bash
python scripts/pretrain_jepa.py \
  --config configs/jepa_t5_encoder_decoder.yaml \
  --model-name-or-path google/mt5-small \
  --train-file data/scenic/SCENIC_full_training_dataset.json \
  --max-steps 1000
```

Use a local checkpoint path in place of `google/mt5-small` for offline or H20
runs.

## Outputs

```text
runs/jepa-mt5-small-scenic/
  jepa_metrics.csv
  jepa_run_config.json
  checkpoint/text_jepa.pt
  checkpoint/tokenizer/
```

## Learning Experiments

Start small:

1. Train JEPA for 100-500 steps and inspect whether latent prediction loss falls.
2. Freeze the context encoder and train a small probe for action/device/room
   labels if those labels are available.
3. Fine-tune the same encoder-decoder model on SCENIC and compare with no-JEPA
   initialization.
4. Compare against the existing decoder-only SFT and contrastive SFT runs.
5. Add T5/ChatLM-mini baselines under the same train/eval split.

## Publishable Claims To Test

- JEPA pretraining improves few-shot SCENIC command generation.
- JEPA embeddings cluster semantically equivalent commands better than causal
  LM or contrastive SFT embeddings.
- JEPA initialization improves pruning retention at 30% and 50% sparsity.
- Encoder-decoder JEPA is more robust to paraphrase and indirect commands than
  direct SFT at the same parameter budget.
