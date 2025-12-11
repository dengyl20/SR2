# Hierarchical Reasoning Model Analysis
This repository contains the code used to analyse the methods proposed in the [HRM paper](https://arxiv.org/abs/2506.21734). This repository is based on the [official HRM repo](https://github.com/sapientinc/HRM), and only adds bits and pieces to run the experiments we performed to (i) replicate the author's results and (ii) understand what drives the performance on ARC-AGI. We document our findings in our [blog post](http://arcprize.org/blog/hrm-analysis). 

## Setup
We have been in close contact with the authors to replicate their experiments. Below is the recipe to create an environment that replicates their results. We have run these on two nodes with 8 H100 GPUs each. We assume CUDA 12.8 is installed.

Step 1: set up a venv.
```bash
sudo snap install astral-uv --classic
uv venv .venv -p 3.12
source .venv/bin/activate
```

Step 2: install dependencies for flash attention
```bash
sudo apt install python3-dev -y
```

Step 3: install pytorch
```bash 
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
uv pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL
```

Step 4: install dependencies for adam-atan2
```bash
uv pip install packaging ninja wheel setuptools setuptools-scm
```

Step 5: install adam-atan2
```bash
uv pip install --no-cache-dir --no-build-isolation adam-atan2 
```
Step 5.1: test if torch, cuda and adam atan2 work
```python
import torch
t = torch.tensor([0,1,2]).to('cuda')
from adam_atan2 import AdamATan2
```

Step 6: install flash attention.  


For Hopper GPUS, install as follows (takes a moment)
```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
cd ../../
```
For Ampere or earlier GPUs, install FlashAttention 2

```bash
uv pip install flash-attn
```

Step 7: install the remaining dependencies
```bash
uv pip install -r requirements.txt
```

### W&B Integration 

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```


## Dataset Preparation
The raw data to replicate our results are public concept-arc and ARC-AGI-v1 data, they are in kaggle/input. To compile the data, run:
```bash
# Dataset Preparation
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/input \
  --output-dir data/arc-aug-1000 \
  --subsets concept training evaluation \
  --test-set-name evaluation
```
## Training

### Replication of ARC-AGI v1 Public Eval
To replicate the public eval results, just run training. On a single node with 8 GPUs, this becomes:
```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py
```
Our replications have been done on 2 nodes with 8 GPUs each. To replicate that, run:
```bash
torchrun \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --nproc_per_node $GPUS_PER_NODE \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  pretrain.py  
```


### Change architecture to transformer

To evaluate the hierarchical architecture, we replace the HRM model with standard transformers, but keep the remaining interfaces and outer refinement loop. The implementation of the transformer can be found in `models/hrm/hrm_act_v2.py`, the hrm model in `models/hrm/hrm_act_v1.py`. To make comparison easy, the transformer is plugged into the H-module, with hierarchical computation removed. The base configuration uses `config/arch/hrm_v1.yaml`.

To test the transformer:
```bash
# Transformer with ACT (adaptive computation)
torchrun --nproc-per-node 8 pretrain.py arch=hrm_v2_params_matched 
```


### Change inner loop steps
The base model uses 2 H_cycles and 2 L_cycles (hierarchical cycles within each reasoning step). To test the effect of these cycles, modify the architecture configuration. These cycles control the depth of reasoning within each adaptive computation step.

```bash
# Test different cycle depths (default: H_cycles=2, L_cycles=2)
torchrun --nproc-per-node 8 pretrain.py arch.H_cycles=1 arch.L_cycles=1  # one call each
torchrun --nproc-per-node 8 pretrain.py arch.H_cycles=2 arch.L_cycles=2  # baseline
torchrun --nproc-per-node 8 pretrain.py arch.H_cycles=4 arch.L_cycles=4  # four cycles each
```

### Change outer loop steps; evaluate with changed outer loop steps
The base model uses outer refinement loops with up to 8 steps and ACT. To test different configurations:

**Training with different ACT settings:**
```bash
# Disable ACT - fixed computation (default uses ACT with halt_max_steps=8)
torchrun --nproc-per-node 8 pretrain.py arch.halt_max_steps=16 +arch.act_enabled=false

# Change maximum ACT steps
torchrun --nproc-per-node 8 pretrain.py arch.halt_max_steps=1   # Single step
torchrun --nproc-per-node 8 pretrain.py arch.halt_max_steps=4   # 4 max steps
torchrun --nproc-per-node 8 pretrain.py arch.halt_max_steps=16  # 16 max steps
```

**Evaluating with different inference steps:**
After training, evaluate the same model with different numbers of inference steps. Create a simple script to test different step counts:

```python
# save as eval_inference_steps.py
from evaluate_trained_model import evaluate_checkpoint

checkpoint = 'checkpoints/your_model/step_XXXXX'  # Update with your checkpoint
data_path = 'data/arc-aug-1000'  # Your test dataset

# Test different max steps
for steps in [1, 2, 4, 8, 16, 32, 64]:
    print(f"Evaluating with {steps} max steps...")
    config_overrides = {
        'arch': {'halt_max_steps': steps},
        'global_batch_size': 512
    }
    evaluate_checkpoint(
        checkpoint_path=checkpoint,
        data_path=data_path,
        output_dir=f'eval_results/steps_{steps}',
        config_overrides=config_overrides
    )
```

Then run with: `torchrun --nproc-per-node 8 eval_inference_steps.py`

### Change augmentations; evaluate with different augmentations
The base model trains with augmented data (default uses `data/arc-aug-1000`). To test different augmentation settings:

**Training with different augmentation counts:**
```bash
# First, build datasets with different augmentation levels
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/input \
  --output-dir data/arc-aug-600 \
  --num-aug 600 \
  --subsets concept training evaluation \
  --test-set-name evaluation

# Train with the new dataset
torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-aug-600
```

**Evaluating with different augmentation counts at inference:**
After training a model, vary how many augmentations are used at inference time. Note that we use random subsampling of the available original+augmented predictions, since changing augmentations at inference time isn't possible with the current hrm architecture.
```bash
# Evaluate the same trained model with varying augmentation counts
python run_augmentation_ablation_eval.py \
  --checkpoint checkpoints/your_model/step_XXXXX \
  --data-path data/arc-aug-1000 \
  --augmentation-counts 1,2,3,5,10,20,40,60,100,200,400,600 \
  --output-dir augmentation_eval_results \
  --random-seed 42 \
  --batch-size 512

# For multi-GPU evaluation (recommended):
torchrun --nproc-per-node 8 run_augmentation_ablation_eval.py \
  --checkpoint checkpoints/your_model/step_XXXXX \
  --data-path data/arc-aug-1000 \
  --augmentation-counts 1,2,3,5,10,20,40,60,100,200,400,600 \
  --output-dir augmentation_eval_results
```
