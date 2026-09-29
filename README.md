# QLoRA Support Router in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garg-shiv/qlora-support-router-colab/blob/main/My_First_LLM_Fine_Tuning.ipynb)

A beginner-friendly, end-to-end example of fine-tuning `Qwen/Qwen2.5-0.5B-Instruct` for customer-support routing with 4-bit QLoRA in Google Colab.

## What the notebook demonstrates

- Loading an instruction model in 4-bit NF4 format
- Building conversational prompt-completion data
- Measuring a base-model baseline
- Applying LoRA adapters with PEFT
- Fine-tuning with TRL's `SFTTrainer`
- Evaluating held-out and boundary cases
- Saving the trained adapter locally or to Google Drive

The classifier returns one of five categories as strict JSON:

```json
{"category": "billing"}
```

Supported categories are `billing`, `technical`, `account`, `cancellation`, and `shipping`.

## Run it

1. Open `My_First_LLM_Fine_Tuning.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the notebook from top to bottom.
4. Optionally enable the final Google Drive cell to persist the LoRA adapter.

The notebook was tested with a Tesla T4 and the pinned packages in `requirements.txt`.

## Repository notes

Model checkpoints and adapter weights are intentionally ignored by Git. Store trained weights in Google Drive or publish them separately to a model registry such as Hugging Face Hub.

The Colab badge is configured for the GitHub account `garg-shiv`.

## Limitations

This is a teaching project with a small synthetic dataset. High validation accuracy here does not establish production readiness. A real system needs representative labeled data, a larger untouched test set, error analysis, safety review, and monitoring.
