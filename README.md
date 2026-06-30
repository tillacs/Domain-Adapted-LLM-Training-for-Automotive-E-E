# Domain-Adapted LLM Training for Automotive E/E

This project is an end-to-end demonstration of adapting a small open-source LLM to the automotive electrical/electronic (E/E) domain using only public data and free compute in a toy scenario. It includes implementation decisions and limitations.


## Motivation

General-purpose language models are pretrained on broad data corpora where automotive E/E related data is structurally underrepresented. This creates two distinct gaps (McCormick 2025):

#### Domain Knowledge Gap
Many automotive E/E terms carry a different meaning than in general language: `bus` refers to a communication network, not a vehicle and core domain concepts like `CAN` arbitration rules or `ASIL` classification are absent or sparse in general pretraining data.

#### Out-of-Distribution (OOD) Text Gap
E/E documentation follows strict formatting conventions, such as hex-encoded `UDS` messages or `ARXML` schemas, that are rarely seen in web text. When format and style fall outside the model's training distribution, it might assign low probability to the tokens.

The pipeline below is built to close both gaps.


## Approach

The project follows a **Base model → CPT → SFT → Evaluation** pipeline, which separates *domain knowledge* from *task behavior*:

- **CPT (Part 2)** continues next-token training on raw E/E text.
- **SFT (Part 3)** then teaches the instruction-response format on top.

The Base variant is used rather than the Instruct variant of the chosen model, because running CPT on unstructured text could degrade the instruction-following an instruct model already has (McCormick 2025).


## Overview

| Part | Task | Deliverable |
|------|------|-------------|
| 1 | Data corpus construction | Strategy write-up + corpus assembly script |
| 2 | Continued Pre-Training (CPT) | Design write-up + training notebook |
| 3 | Supervised Fine-Tuning (SFT) | 20 instruction pairs (JSONL) + SFT write-up |
| 4 | Evaluation | Eval plan + eval questions with reference answers |

All steps use only public data and free compute.


## Model Choice

**Llama-3.2-1B** has been chosen as the starting point. The notebooks use the openly hosted `unsloth/Llama-3.2-1B` (and its 4-bit variant `unsloth/Llama-3.2-1B-bnb-4bit`). This is an ungated re-host of Meta's `meta-llama/Llama-3.2-1B`.

| Property           | Value                         |
| ------------------ | ----------------------------- |
| Publisher          | Meta (2024)                   |
| Parameters         | 1.24B                         |
| Tokenizer          | TikToken-based                |
| Context window     | 128k tokens                   |

#### Why decoder-only?

The goal is a generative E/E assistant that answers open questions, so the model has to generate. Decoder-only models are autoregressive and built for that. An encoder model classifies or retrieves and cannot generate open answers, so it does not fit open-ended Q&A.

#### Why Llama-3.2-1B?

At 1.24B parameters the model fits a free Kaggle T4 (15 GB VRAM) and is available as a pre-quantized 4-bit checkpoint. This makes it well-suited to a toy setup where memory efficiency and quick iteration matter more than maximal capability.

#### Why Unsloth?

[Unsloth](https://github.com/unslothai/unsloth) is the training framework used for both CPT and SFT. It provides optimized LoRA/QLoRA kernels that roughly halve VRAM use and speed up training, pre-quantized 4-bit checkpoints (`unsloth/Llama-3.2-1B-bnb-4bit`) and a one-call merged 16-bit export (`save_pretrained_merged`), which is used later.


## Prerequisites

**Part 1 (corpus script):** runs locally on CPU. Install the dependencies:
```bash
pip install -r requirements.txt
```

**Parts 2 and 3 (notebooks):** run on Kaggle with a free T4 GPU. Each notebook installs its own dependencies in the first cell and pulls the corpus and instruction pairs from the project's GitHub repo, so no local setup is needed beyond a Kaggle account with GPU enabled.


## Running the Notebooks

The CPT- and the SFT-notebook run on Kaggle (free T4 GPU) and save a **merged 16-bit model** to the Kaggle working directory at the end:

```python
model.save_pretrained_merged(OUTPUT, tokenizer, save_method="merged_16bit")
```
To use the trained model in a following notebook, attach the output as an input.
Kaggle then mounts it under /kaggle/input/... Set that path in the config:

```python
CPT_MODEL_PATH = "/kaggle/input/<cpt-notebook-slug>/cpt_merged"
```

## AI Assistance

Claude Opus 4.8 (Anthropic) was used as a development assistant for research, debugging and scaling.

## Repository Structure

```
Domain-Adapted-LLM-Training-for-Automotive-E-E/
├── README.md
├── requirements.txt                # Part 1 dependencies (notebooks install their own)
├── part1_data/
│   ├── corpus_strategy.md          # data corpus strategy write-up
│   ├── data_corpus_script.py       # corpus collection + filtering + chunking script
│   └── data_corpus/                # script output: train.json + val.json + overview
│       ├── train.json              # tokenized training chunks
│       ├── val.json                # tokenized validation chunks
│       └── corpus_stats.csv        # per-document metrics + filter status
├── part2_cpt/
│   ├── cpt_design.md               # CPT design write-up
│   └── cpt_notebook.ipynb          # CPT training notebook (Kaggle T4)
├── part3_sft/
│   ├── sft_approach.md             # SFT approach write-up
│   ├── instruction_pairs.jsonl     # 20 instruction/response pairs
│   └── sft_notebook.ipynb          # SFT training notebook (Kaggle T4)
└── part4_eval/
    └── eval_plan.md                # evaluation plan + eval questions
```


## References

- McCormick, C. (2025). *Continuing Pre-Training on Raw Text.* mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/