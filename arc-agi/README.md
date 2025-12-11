# SR2 Evaluation on ARC-AGI (ARC-1 & ARC-2)

[![Python](https://img.shields.io/badge/Python-3.12-informational)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-informational)](https://developer.nvidia.com/cuda-zone)
[![ARC-AGI](https://img.shields.io/badge/Benchmark-ARC--1%20%26%20ARC--2-orange)](https://github.com/arcprize/hierarchical-reasoning-model-analysis)

This repository contains the code used to evaluate **SR2** on the **ARC-AGI** benchmarks, specifically **ARC-1** and **ARC-2**.

To ensure a fair comparison with prior work and to reuse the official tooling, this project is implemented as a targeted modification of the ARC Prize repository  
[arcprize/hierarchical-reasoning-model-analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis).  
We replace the original **HRM** model and its training pipeline with **SR2**, while keeping the surrounding infrastructure (layers, optimizer, loss design, evaluation utilities, etc.) unchanged.

---

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training & Evaluation](#training--evaluation)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Logs & Checkpoints](#logs--checkpoints)
- [Environment & Performance Notes](#environment--performance-notes)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository provides:

- An **SR2-based implementation** within the ARC Prize hierarchical reasoning framework.
- A **training and evaluation pipeline** for running SR2 on **ARC-1** and **ARC-2**.
- **Instructions to reproduce** the results reported in our SR2 paper on these benchmarks.
- **Reference artifacts**, including:
  - Weights & Biases (wandb) training logs/reports.
  - Final and best **checkpoints** for SR2 on ARC-1 and ARC-2.

---

## Key Components

The main SR2-related components are:

- `pretrain_arc.py`  
  Training script adapted to **pretrain SR2** on ARC-AGI, using an **EMA (Exponential Moving Average)** of model parameters for more stable evaluation.

- `models/hrm/sr2.py`  
  The SR2 model implementation, integrated into the existing HRM codebase so that it can be used transparently in place of the original HRM model.

- `config/cfg_pretrain.yaml`  
  Central configuration file controlling:
  - Dataset paths and benchmark selection (ARC-1 vs ARC-2).
  - Training hyperparameters.
  - Logging and checkpointing behavior.

- Shell utilities:
  - `prepare_dataset.sh` – prepares and compiles the ARC-1 and ARC-2 datasets.
  - `train_and_eval.sh` – runs the end-to-end training and evaluation pipeline.

> [!NOTE]
> The core infrastructure (layers, optimizer, loss functions, evaluation logic, etc.) follows the original  
> `arcprize/hierarchical-reasoning-model-analysis` implementation to maintain comparability with HRM.

---

## Getting Started

### Prerequisites

- **Operating system**: Linux (recommended for CUDA and FlashAttention support).
- **Python**: 3.12 (reference environment).
- **CUDA**: 12.8 (reference environment).
- **GPU**: Experiments were run with **8 × NVIDIA H200 NVL** GPUs.
- **FlashAttention**: FlashAttention **v3** (see notes below on compatibility).

> [!IMPORTANT]
> Different combinations of **CUDA** and **flash-attn** versions can affect both **training stability** and **throughput**.  
> For the closest reproduction of our results, we recommend matching the reference environment as closely as possible.

### Installation

1. Clone this repository:

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Some specialized dependencies (including those related to FlashAttention and certain low-level kernels) are inherited from the original ARC Prize repository.
   For details, please also refer to the upstream documentation:

   * `hrm_analysis.md` in the original repo
     [arcprize/hierarchical-reasoning-model-analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis)

> [!TIP]
> If you encounter build or installation issues related to `flash-attn`, first verify that your CUDA toolkit,
> driver version, and PyTorch build are mutually compatible, and then follow the upstream instructions in `hrm_analysis.md`.

---

## Data Preparation

We use the same ARC-AGI data preparation pipeline as the upstream HRM repository.

1. Run the dataset preparation script:

   ```bash
   bash prepare_dataset.sh
   ```

2. This script will download/compile the **ARC-1** and **ARC-2** datasets into the expected directory layout for training and evaluation.

> [!NOTE]
> The exact output directory structure depends on `prepare_dataset.sh`.
> In the following sections, we refer to the root of the prepared dataset as `<DATA_ROOT>`.

---

## Configuration

All major configuration options are defined in:

```text
config/cfg_pretrain.yaml
```

Key settings include:

* `data_path`:
  Root path to the ARC dataset to use. Point this to either the ARC-1 or ARC-2 dataset produced by `prepare_dataset.sh`.

  Examples:

  ```yaml
  # For ARC-1
  data_path: /path/to/<DATA_ROOT>/arc_1

  # For ARC-2
  data_path: /path/to/<DATA_ROOT>/arc_2
  ```

* Training-related parameters (e.g., batch size, learning rate, number of steps).

* Logging and checkpoint paths (e.g., output directories, wandb settings).

> [!TIP]
> To switch between **ARC-1** and **ARC-2** experiments, you typically only need to adjust `data_path`
> (and any dataset-specific configuration, if applicable) in `cfg_pretrain.yaml`.

---

## Training & Evaluation

The recommended entry point for running SR2 on ARC-AGI is:

```bash
bash train_and_eval.sh
```

By default, this script will:

1. Load configuration from `config/cfg_pretrain.yaml`.
2. Launch SR2 training using `pretrain_arc_ema.py`.
3. Evaluate the trained model on the chosen benchmark (**ARC-1** or **ARC-2**), using the official ARC evaluation tooling from the ARC Prize repository.

> [!NOTE]
> `train_and_eval.sh` is intended as a single entry point for end-to-end experiments
> (from training through evaluation). For advanced use cases, you can inspect and modify the script
> to customize launch parameters, distributed settings, or logging options.

---

## Reproducing Paper Results

This repository is designed to reproduce the SR2 results reported on **ARC-1** and **ARC-2** in our paper.

A typical workflow is:

1. **Prepare the environment**

   * Match the reference environment as closely as possible:

     * 8 × H200 NVL
     * CUDA 12.8
     * Python 3.12
     * FlashAttention 3

2. **Prepare the data**

   ```bash
   bash prepare_dataset.sh
   ```

3. **Select the benchmark**

   * Edit `config/cfg_pretrain.yaml` and set:

     * `data_path` → ARC-1 or ARC-2 dataset root (as produced by `prepare_dataset.sh`).

4. **Run training and evaluation**

   ```bash
   bash train_and_eval.sh
   ```

5. **Compare with reference artifacts**

   * Use the provided **wandb log reports** and **best checkpoints** (see below) to:

     * Verify that training curves and final metrics are aligned.
     * Confirm that your run matches the reported performance within expected variance.

> [!TIP]
> If you only want to **re-evaluate** a released checkpoint (without re-training SR2),
> you can adapt `train_and_eval.sh` or directly call the evaluation logic used there,
> pointing it to the corresponding checkpoint file.

---

## Logs & Checkpoints

We provide both **training logs** and **final/best checkpoints** to facilitate reproducibility.

### Weights & Biases Logs

The SR2 training process is logged to **Weights & Biases (wandb)**, including:

* Training loss
* Validation metrics
* ARC-1 / ARC-2 evaluation scores

In this repository (or its release assets), you will find exported wandb reports corresponding to:

* SR2 on **ARC-1**
* SR2 on **ARC-2**

You can use these as a reference when validating your own runs.

### Best Checkpoints

The best-performing SR2 checkpoints on ARC-1 and ARC-2 are provided as reference.

A suggested way to present them in this README is:

| Benchmark | Log / Report (wandb or export)              | Best Checkpoint Path / Link                 |
| --------: | ------------------------------------------- | ------------------------------------------- |
|     ARC-1 | [wandb Report](https://api.wandb.ai/links/sayagugu-mbzuai/8n0iv1si) | [HF Checkpoints](https://huggingface.co/SayaGugu/SR2/tree/main/Arc-aug-1000%20ACT-torch)  |
|     ARC-2 | [wandb Report](https://api.wandb.ai/links/sayagugu-mbzuai/x47llwlo) | [HF Checkpoints](https://huggingface.co/SayaGugu/SR2/tree/main/Arc-2-aug-1000%20ACT-torch) |


Replace the placeholders above with the actual file paths or external links (e.g., release assets, wandb run URLs).

> [!IMPORTANT]
> When using these checkpoints, please ensure that your configuration
> (model architecture, tokenizer/preprocessing, etc.) matches the setup used to train them.

---

## Environment & Performance Notes

* All SR2 experiments in this repository were originally run on:

  * **8 × H200 NVL GPUs**
  * **CUDA 12.8**
  * **Python 3.12**
  * **FlashAttention 3**

* While SR2 should run on other GPU setups and CUDA versions, you may observe differences in:

  * Training throughput
  * Numerical stability
  * Final benchmark performance

* In particular, different combinations of:

  * `flash-attn` versions
  * CUDA toolkit versions
  * GPU architectures

  can subtly change training dynamics and evaluation scores.

> [!NOTE]
> If you deviate from the reference environment and observe unexpected performance,
> please check:
>
> * That FlashAttention is correctly installed and actually used.
> * That your mixed-precision settings (e.g., `bf16`/`fp16`) match the configuration.
> * That you have not altered key hyperparameters in `cfg_pretrain.yaml`.

---

## Acknowledgements

This work builds directly on the excellent infrastructure and ARC evaluation tooling provided by the
[ARC Prize](https://arcprize.org/) team and their repository:

* [arcprize/hierarchical-reasoning-model-analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis)

We thank the authors and maintainers of that project for making their code and benchmarks available,
which enabled a clean and fair comparison between **HRM** and **SR2** on **ARC-1** and **ARC-2**.


