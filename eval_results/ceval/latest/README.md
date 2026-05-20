# Latest C-Eval Result

Decoder-only 0.2B checkpoint evaluated on C-Eval validation with 5-shot prompting.

| Field | Value |
| --- | --- |
| Checkpoint | `runs/h20-8gpu-llama-0p2b-deepspeed/latest` |
| Dataset | `ceval/ceval-exam` |
| Split | `val` |
| Subjects | 52 |
| Shots | 5 |
| Chat format | `true` |
| Normalize by length | `false` |
| Correct / Total | 320 / 1346 |
| Accuracy | 23.77% |

## Category Split

| category | correct | question_count | accuracy |
| --- | ---: | ---: | ---: |
| Humanities | 67 | 257 | 26.07% |
| Other | 89 | 384 | 23.18% |
| STEM | 90 | 430 | 20.93% |
| Social Science | 74 | 275 | 26.91% |

## Best Subjects

| Subject | Correct / Total | Accuracy |
| --- | ---: | ---: |
| `art_studies` | 15 / 33 | 45.45% |
| `discrete_mathematics` | 6 / 16 | 37.50% |
| `high_school_biology` | 7 / 19 | 36.84% |
| `civil_servant` | 17 / 47 | 36.17% |
| `education_science` | 10 / 29 | 34.48% |

Raw files:

- [`ceval_summary.json`](ceval_summary.json)
- [`ceval_category_summary.csv`](ceval_category_summary.csv)
- [`ceval_predictions.csv`](ceval_predictions.csv)
