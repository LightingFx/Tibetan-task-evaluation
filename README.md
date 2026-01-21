# Tibetan Task Evaluation

A comprehensive toolkit for evaluating Large Language Models (LLMs) on Tibetan language tasks using `vLLM` and `transformers`.

## Overview

This repository contains scripts to evaluate models on various benchmarks, including:
- **TLUE (Tibetan Language Understanding Evaluation)**: General language understanding.
- **GSM8K-BO**: Tibetan translation of the GSM8K math reasoning dataset.
- **Belebele**: Machine reading comprehension.
- **Vocabulary Efficiency**: Metrics for tokenization efficiency (CpT, BPT).

## Requirements

- Python 3.8+
- `vllm`
- `transformers`
- `torch`
- `numpy`
- `tqdm`

## Usage

### 1. TLUE Evaluation (`eval_tlue_vllm.py`)

Evaluates models on the TLUE benchmark with MMLU-style categorization.

```bash
python eval_tlue_vllm.py \
    --model_path /path/to/your/model \
    --data_path /path/to/tlue/data \
    --n_shot 0 \
    --gpu_memory_utilization 0.9
```

**Key Arguments:**
- `--model_path`: Path to the HF model.
- `--n_shot`: Number of few-shot examples (default: 0).
- `--gpu_memory_utilization`: vLLM GPU memory usage limit.

### 2. GSM8K Tibetan Evaluation (`eval_gsm8k-bo.py`)

Evaluates mathematical reasoning using the GSM8K dataset translated into Tibetan.

```bash
python eval_gsm8k-bo.py \
    --model_path /path/to/your/model \
    --data_path eval_data/gsm8k_bo.json \
    --prompt_type direct \
    --n_shot 4
```

**Key Arguments:**
- `--prompt_type`: `direct` (answer only) or `cot` (chain-of-thought).
- `--n_shot`: Number of few-shot examples.

### 3. Belebele Evaluation (`eval_bele.py`)

Evaluates reading comprehension capabilities.

```bash
python eval_bele.py \
    --model /path/to/your/model \
    --data_path /path/to/belebele.json
```

### 4. Vocabulary Efficiency (`eval_vocab.py`)

Calculates Characters per Token (CpT) and Bits per Token (BPT) to measure tokenizer efficiency for Tibetan.

```bash
python eval_vocab.py \
    --model_name_or_path /path/to/your/model \
    --data_path /path/to/text/data
```

## Directory Structure

```
.
├── eval_bele.py        # Belebele benchmark evaluation
├── eval_gsm8k-bo.py    # GSM8K (Tibetan) evaluation
├── eval_tlue_vllm.py   # TLUE benchmark evaluation
├── eval_vocab.py       # Vocabulary efficiency metrics
├── eval_data/          # Dataset directory
│   └── gsm8k_bo.json   # GSM8K Tibetan dataset
└── README.md
```
