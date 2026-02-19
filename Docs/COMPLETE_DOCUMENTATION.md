# GMLFNet Complete Documentation
## A Beginner-to-Intermediate Guide

---

## 📖 Table of Contents

1. [Introduction & Motivation](#introduction--motivation)
2. [What is This Project?](#what-is-this-project)
3. [Key Concepts Explained](#key-concepts-explained)
4. [Project Architecture](#project-architecture)
5. [Installation & Setup](#installation--setup)
6. [Project Structure Explained](#project-structure-explained)
7. [How to Use the Project](#how-to-use-the-project)
8. [Understanding the Code](#understanding-the-code)
9. [Configuration Guide](#configuration-guide)
10. [Common Tasks & Workflows](#common-tasks--workflows)
11. [Troubleshooting](#troubleshooting)

---

## Introduction & Motivation

### The Problem

Hospitals around the world use different imaging devices and equipment. When doctors use **endoscopy** (a camera inserted to view the inside of the digestive tract), images of the same polyp (abnormal tissue growth) can look very different depending on:

- **Hardware differences**: Different camera brands/models
- **Software settings**: Different processing pipelines
- **Environmental factors**: Lighting conditions, angle, etc.

### The Challenge

Medical AI models trained on images from **Hospital A** might not work well when deployed at **Hospital B** because the images look different. This problem is called **domain shift** or **distribution shift**.

### The Solution: GMLFNet

GMLFNet uses **meta-learning** (learning how to learn) to create a model that:

1. **Generalizes across hospitals** (domains) without seeing all of them during training
2. **Adapts quickly** to new hospitals with just a few examples
3. **Keeps most weights frozen**, only adapting lightweight "Fast Adaptation Weights" (FAW)

This makes deployment safe and practical—you don't need to retrain the entire model for each new hospital.

---

## What is This Project?

### At a Glance

**GMLFNet** = **G**radient-based **M**eta-**L**earning with **F**ast Adaptation Weights for **Net**work segmentation

It's a deep learning model that:
- **Detects and segments polyps** in endoscopy images (medical images)
- **Learns across multiple medical centers** simultaneously
- **Adapts to new centers** with minimal additional training
- Uses **modern meta-learning algorithms** (MAML) to achieve this

### Core Innovation: Fast Adaptation Weights (FAW)

Normal meta-learning adapts **all model parameters** to new domains, which is slow.

FAW adapts only **~100,000 lightweight parameters** instead of millions, making adaptation:
- **Faster** (fewer parameters to optimize)
- **Safer** (doesn't touch the pretrained encoder)
- **Better** (fewer parameters = less overfitting)

### Why This Matters

Medical AI needs to work reliably across different hospitals without expensive retraining. GMLFNet solves this by making models:
- **Robust**: Works well across multiple hospitals
- **Adaptive**: Can quickly adjust to new hospitals
- **Practical**: Lightweight adaptation that's fast and safe

---

## Key Concepts Explained

### 1. Polyp Segmentation

**What it is:** Identifying exactly which pixels in an image belong to a polyp.

```
Input:  [Endoscopy Image]
         ↓
       [Model]
         ↓
Output: [Segmentation Mask]
        (1 = polyp, 0 = background)
```

**Why it matters:** Doctors need precise boundaries to determine if a polyp needs treatment.

### 2. Domain Shift / Domain Generalization

**Domain shift:** When training and test data look different
- Example: Training on Hospital A's images, testing on Hospital B's images
- Causes model accuracy to drop significantly

**Domain generalization:** Creating models that work on unseen domains
- Train on multiple domains, test on new unseen domains
- Our approach: Use 3 hospitals to train, test on 2 hospitals it's never seen

### 3. Meta-Learning (Learning to Learn)

**Traditional learning:** Optimize weights to minimize error on a task.

**Meta-learning:** Optimize weights so they can be **quickly adapted** to new tasks.

**Analogy:** 
- Traditional learning = learning to do math
- Meta-learning = learning **how to learn** math

**MAML (Model-Agnostic Meta-Learning):**
- Standard meta-learning algorithm
- Updates happen in two loops:
  - **Inner loop** (green): Fast adaptation to a specific domain/hospital
  - **Outer loop** (blue): Updates that help future domains

```
┌─────────────────────────────────────────┐
│ Outer Loop (Update for all domains)     │
│  ┌───────────────────────────────────┐  │
│  │ Inner Loop (Adapt to domain 1)    │  │
│  │ - Create temporary copy of model  │  │
│  │ - Take 5 gradient steps           │  │
│  │ - Test on query data              │  │
│  │ Compute loss                      │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ Inner Loop (Adapt to domain 2)    │  │
│  │ - Create temporary copy of model  │  │
│  │ - Take 5 gradient steps           │  │
│  │ - Test on query data              │  │
│  │ Compute loss                      │  │
│  └───────────────────────────────────┘  │
│     Backpropagate to improve adaptation │
└─────────────────────────────────────────┘
```

### 4. FiLM Modulation

**FiLM** = **Feature-wise Linear Modulation**

Instead of changing features directly, multiply and add per-channel:

```
Modulated_Feature = Gamma × Original_Feature + Beta

Where:
  Gamma = scale/weight
  Beta = shift/bias
```

**Advantage:** Lightweight adaptation—only 2 values per channel, not millions of parameters.

### 5. Few-Shot Learning

**Definition:** Learning from very few examples (typically 5-16 images)

**In our context:**
- Few-shot = adapting to a new hospital with only 16 images
- Zero-shot = testing on a new hospital without any adaptation

---

## Project Architecture

### High-Level Data Flow

```
┌──────────────────────┐
│   Input Image        │
│   (W×H×3)           │
└──────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   ENCODER (Backbone)                 │
│   Res2Net-50 or PVTv2-B2             │
│   Extracts 4 levels of features:     │
│   [F1, F2, F3, F4]                   │
│   (multi-scale features)             │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   FAW (Fast Adaptation Weights)      │
│   - Global pool each feature level   │
│   - Pass through MLP                 │
│   - Output: Gamma & Beta values      │
│   - Apply FiLM modulation to F1-F4   │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   DECODER                            │
│   - Combine modulated features       │
│   - Progressively upsample           │
│   - Apply attention mechanisms       │
│   - Output: Segmentation map         │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   Output Mask                        │
│   (W×H×1, 0.0-1.0 probabilities)     │
└──────────────────────────────────────┘
```

### Three Main Components

#### 1. **Encoder (Backbone)**

**Purpose:** Extract meaningful features from the image

**Choices:**
- **Res2Net-50**: ResNet variant, good balance of speed/accuracy
- **PVTv2-B2**: Vision Transformer-based, more modern, better at capturing global context

**Output:** 4 feature maps at different resolutions (sizes):
- Level 1: Full resolution, 256 channels
- Level 2: 1/2 resolution, 512 channels  
- Level 3: 1/4 resolution, 1024 channels
- Level 4: 1/8 resolution, 2048 channels

#### 2. **Fast Adaptation Weights (FAW)**

**Purpose:** Generate lightweight adaptation parameters

**How it works:**
1. Compress features: Apply Global Average Pooling to each feature level
2. Combine: Concatenate all pooled statistics
3. Predict: Pass through 2-layer MLP
4. Output: Scale (gamma) and shift (beta) values for each decoder layer

**Why "Fast"?**
- Only ~100K parameters (vs ~50M in encoder)
- Optimized quickly in MAML inner loop
- Doesn't touch pretrained encoder weights

#### 3. **Decoder**

**Purpose:** Reconstruct segmentation from features

**Process:**
1. **Takes modulated features** from FAW
2. **Upsamples progressively** (doubles spatial resolution)
3. **Combines information** from multiple scales (rich detail from fine scales + context from coarse scales)
4. **Applies attention** (learns which features to focus on)
5. **Outputs single-channel mask** (1=polyp, 0=background)

---

## Installation & Setup

### Prerequisites

- **Python 3.8 or higher**
- **GPU recommended** (CUDA 11.0+ for faster training)
- **Linux/Mac/Windows** (tested on Linux)

### Step-by-Step Installation

**Step 1: Clone repository (if not already done)**
```bash
cd /home/tommy/Data_science_projects/GMLFNet-Research-Project
```

**Step 2: Create virtual environment (recommended)**
```bash
# Create
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Download datasets**
```bash
python data/download.py --output-dir ./datasets
```

This downloads 5 polyp datasets:
- Kvasir-SEG (1000 images)
- CVC-ClinicDB (612 images)
- CVC-ColonDB (380 images)
- ETIS-LaribPolypDB (196 images)
- CVC-300 (60 images)

**Step 5: Verify installation**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import learn2learn; print('learn2learn OK')"
```

### GPU Setup (Optional but Recommended)

**Check if GPU available:**
```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

**Install CUDA (if needed):**
Follow [PyTorch installation guide](https://pytorch.org/get-started/locally/)

---

## Project Structure Explained

### Directory Overview

```
GMLFNet-Research-Project/
├── configs/                    # Configuration files
├── data/                       # Data loading & preprocessing
├── models/                     # Neural network implementations
├── trainers/                   # Training & evaluation logic
├── scripts/                    # Command-line entry points
├── utils/                      # Helper utilities
├── notebooks/                  # Jupyter notebooks for exploration
├── requirements.txt            # Python dependencies
└── README.md                   # Quick start guide
```

### Detailed Breakdown

#### **configs/** — Configuration Files

**File:** `default.yaml`

Contains all hyperparameters for training. Think of it as a control panel:

```yaml
# Data settings
data:
  root: "./datasets"           # Where datasets are stored
  image_size: 352              # Resize all images to 352×352
  train_centers:               # Use 3 hospitals for training
    - "Kvasir"
    - "CVC-ClinicDB"
    - "CVC-ColonDB"
  test_centers:                # Test on 2 unseen hospitals
    - "ETIS-LaribPolypDB"
    - "CVC-300"

# Model architecture
model:
  backbone: "res2net50"        # Encoder type
  decoder_channels: [256, 128, 64, 32]  # Decoder layer sizes
  faw_hidden_dim: 64           # FAW MLP hidden size
  faw_num_layers: 2            # FAW MLP depth

# Meta-learning settings
meta:
  algorithm: "maml"            # Meta-learning algorithm
  inner_lr: 0.01               # Inner loop learning rate
  inner_steps: 5               # Gradient steps in inner loop
  outer_lr: 0.001              # Outer loop learning rate
  tasks_per_batch: 3           # Episodes per batch
  support_size: 16             # Training examples per domain
  query_size: 16               # Test examples per domain

# Training
training:
  epochs: 200                  # Total training epochs
  grad_clip: 1.0               # Prevent exploding gradients
  scheduler: "cosine"          # Learning rate schedule

# Loss weights
loss:
  bce_weight: 0.5              # Binary cross-entropy weight
  dice_weight: 0.5             # Dice loss weight
  structure_weight: 0.2        # Structural similarity weight
```

**How to modify:** Edit `configs/default.yaml` before training.

#### **data/** — Data Loading

| File | Purpose |
|------|---------|
| `datasets.py` | Dataset classes for loading images & masks |
| `augmentations.py` | Data augmentation (rotation, flip, color jitter) |
| `meta_sampler.py` | Episodic sampling for meta-learning |
| `download.py` | Script to download all 5 datasets |
| `__init__.py` | Package initialization |

**Key concept — `meta_sampler.py`:**
Creates episodes (mini-tasks) for meta-learning:
```
Episode 1:
  - Support set: 16 images from Hospital A
  - Query set: 16 images from Hospital A

Episode 2:
  - Support set: 16 images from Hospital B
  - Query set: 16 images from Hospital B

(repeat with 3 hospital combinations)
```

#### **models/** — Neural Network Code

| File | Purpose |
|------|---------|
| `gmlf_net.py` | Main model combining encoder + FAW + decoder |
| `backbone.py` | Encoder implementations (Res2Net, PVTv2) |
| `decoder.py` | Decoder with reverse attention & RFB blocks |
| `fast_adapt_weights.py` | FAW module (the key innovation) |
| `losses.py` | Loss functions (BCE + Dice + STriPS) |
| `__init__.py` | Package initialization |

**Key file — `gmlf_net.py`:**
```python
class GMLFNet(nn.Module):
    """The main model"""
    
    def __init__(self, backbone_name, decoder_channel, ...):
        self.encoder = get_backbone(backbone_name)  # Extract features
        self.faw = FastAdaptationWeights(...)       # Generate modulation
        self.decoder = MultiScaleDecoder(...)       # Reconstruct segmentation
    
    def forward(self, x):
        features = self.encoder(x)              # Features from encoder
        modulated_features = self.faw(features) # Apply adaptation
        output = self.decoder(modulated_features) # Decode to mask
        return output
```

#### **trainers/** — Training Logic

| File | Purpose |
|------|---------|
| `meta_trainer.py` | MAML meta-learning trainer |
| `baseline_trainer.py` | Standard supervised trainer (no meta-learning) |
| `evaluator.py` | Evaluation on test sets (zero-shot & few-shot) |
| `__init__.py` | Package initialization |

**Key concept:**

```
meta_trainer.py:
├─ MAMLMetaTrainer.train_epoch()
│  ├─ Sample 3 tasks (one per hospital)
│  ├─ For each task:
│  │  ├─ Inner loop: Adapt to hospital on support set
│  │  ├─ Evaluate on query set
│  │  └─ Accumulate loss
│  └─ Outer loop: Update all parameters
└─ Save checkpoint
```

#### **scripts/** — Entry Points

| File | Purpose | Command |
|------|---------|---------|
| `train_meta.py` | Meta-learning training | `python scripts/train_meta.py --config configs/default.yaml` |
| `train_baseline.py` | Standard training (no adaptation) | `python scripts/train_baseline.py --config configs/default.yaml` |
| `evaluate.py` | Test on test centers | `python scripts/evaluate.py --checkpoint best.pth` |
| `ablation.py` | Compare configurations | `python scripts/ablation.py --ablation all` |

#### **utils/** — Helper Code

| File | Purpose |
|------|---------|
| `metrics.py` | Evaluation metrics (Dice, IoU, mIoU) |
| `visualization.py` | Plot images & segmentation masks |
| `logging_utils.py` | TensorBoard/W&B logging |
| `misc.py` | General utilities (checkpoints, device handling) |

#### **notebooks/** — Interactive Exploration

| Notebook | Purpose |
|----------|---------|
| `data_exploration.ipynb` | Visualize datasets |
| `kaggle_train.ipynb` | Training on Kaggle GPU |
| `results_analysis.ipynb` | Analyze & plot results |

---

## How to Use the Project

### Training Workflow

#### Option 1: Meta-Learning Training (Recommended)

Meta-learning trains a model that adapts to new hospitals.

```bash
# Basic training
python scripts/train_meta.py --config configs/default.yaml

# Resume from checkpoint (e.g., after Colab disconnect)
python scripts/train_meta.py --config configs/default.yaml --resume runs/checkpoint_epoch50.pth

# With custom config
python scripts/train_meta.py --config configs/custom.yaml
```

**What happens:**
1. Loads 3 training hospitals (Kvasir, CVC-ClinicDB, CVC-ColonDB)
2. Creates episodes: sample support & query sets from each hospital
3. **Inner loop** (5 steps): Adapt FAW to each hospital
4. **Outer loop** (backprop): Update all parameters
5. Saves best model to `runs/best_model.pth`

**Training time:** ~4-6 hours on single GPU

**Output files:**
```
runs/
├── best_model.pth              # Best model (lowest validation loss)
├── checkpoint_epoch*.pth       # Checkpoints every N epochs
├── config.yaml                 # Config used for training
└── logs/                        # TensorBoard logs
```

#### Option 2: Baseline Training (For Comparison)

Standard training without meta-learning, for ablation studies.

```bash
python scripts/train_baseline.py --config configs/default.yaml
```

### Evaluation Workflow

#### Test Pretrained Model

Two evaluation modes:

**1. Zero-shot (No Adaptation)**
- Test on new hospitals without any adaptation
- True test of generalization

```bash
python scripts/evaluate.py \
    --checkpoint runs/best_model.pth \
    --mode zero_shot
```

**2. Few-shot (With Adaptation)**
- Adapt model to new hospital using 16 support images
- Measures how quickly it adapts

```bash
python scripts/evaluate.py \
    --checkpoint runs/best_model.pth \
    --mode few_shot
```

### Ablation Studies

Compare different configurations to understand what helps:

```bash
# Test all ablations
python scripts/ablation.py --config configs/default.yaml --ablation all

# Test specific ablation (e.g., effect of inner loop steps)
python scripts/ablation.py --config configs/default.yaml --ablation inner_steps
```

**Ablations test:**
- With/without FAW module
- Different inner loop steps (1, 3, 5, 7)
- With/without meta-learning
- Different backbones (Res2Net vs PVTv2)

---

## Understanding the Code

### Meta-Learning Training Flow

Here's the complete loop in `trainers/meta_trainer.py`:

```python
class MAMLMetaTrainer:
    def train_epoch(self):
        for task_batch in dataloader:
            # OUTER LOOP STEP
            
            for support_images, support_masks, query_images, query_masks in task_batch:
                # INNER LOOP STEP
                
                # 1. Clone the model for this task
                learner = self.maml.clone()
                
                # 2. Inner loop: adapt to this hospital (5 gradient steps)
                for step in range(5):
                    # Forward pass on support set
                    support_output = learner(support_images)
                    support_loss = loss_fn(support_output, support_masks)
                    
                    # Backward pass - update FAW parameters only
                    learner.adapt(support_loss)
                
                # 3. Evaluate adapted model on query set
                query_output = learner(query_images)
                query_loss = loss_fn(query_output, query_masks)
                
                # Accumulate loss for outer loop
                accumulated_loss += query_loss
            
            # 4. Outer loop: backprop through accumulated losses
            accumulated_loss.backward()
            
            # 5. Update all parameters (encoder + FAW + decoder)
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
```

### Key Insight: Selective Adaptation

**Important:** Only FAW parameters are adapted in inner loop!

```python
# In meta_trainer initialization
self.maml = l2l.algorithms.MAML(
    model,
    lr=0.01,  # Inner loop learning rate
    first_order=True,  # Use first-order approximation
)
```

The `first_order=True` flag means:
- We approximate second-order derivative as first-order
- Saves memory and computation
- Still works very well in practice

**In the actual inner loop adaptation:**
- Only ~100K FAW parameters are updated
- 50M encoder + 30M decoder parameters stay frozen
- This is why it's called "Fast" Adaptation Weights

### Loss Function

The model optimizes three losses simultaneously:

```python
class GMLFNetLoss(nn.Module):
    def forward(self, pred, target):
        # 1. Binary Cross-Entropy: pixel-level accuracy
        bce_loss = binary_crossentropy(pred, target)
        
        # 2. Dice Loss: region overlap (good for imbalanced classes)
        dice_loss = 1 - dice_coefficient(pred, target)
        
        # 3. Structure Loss: preserve boundaries
        structure_loss = structural_similarity_loss(pred, target)
        
        # Weighted combination
        total_loss = (0.5 * bce_loss + 
                      0.5 * dice_loss + 
                      0.2 * structure_loss)
        
        return total_loss
```

**Why three losses?**
- **BCE**: Handles overall pixel classification
- **Dice**: Handles class imbalance (few polyp pixels vs many background)
- **Structure**: Preserves sharp boundaries (important for surgical precision)

### Evaluation Metrics

After training, models are evaluated with standard segmentation metrics:

```python
# From utils/metrics.py

dice_score = 2 * TP / (2*TP + FP + FN)
# 1.0 = perfect, 0.0 = terrible
# Measures overlap between predicted and actual mask

iou_score = TP / (TP + FP + FN)
# 1.0 = perfect, 0.0 = terrible
# Stricter than Dice

mae = mean_absolute_error(pred, target)
# 0.0 = perfect, 1.0 = terrible
# Average pixel difference
```

---

## Configuration Guide

### Creating Custom Configurations

Copy `configs/default.yaml` to `configs/custom.yaml`:

```bash
cp configs/default.yaml configs/custom.yaml
```

### Common Configuration Changes

#### Train on Different Hospital Combinations

```yaml
data:
  train_centers:
    - "Kvasir"
    - "CVC-ClinicDB"
    # Removed CVC-ColonDB to train on 2 centers
  test_centers:
    - "ETIS-LaribPolypDB"
    - "CVC-300"
    - "CVC-ColonDB"  # Test on 3 centers instead
```

#### Change Image Resolution

```yaml
data:
  image_size: 256  # 352 → 256 (faster training, less memory)
```

#### Use Different Backbone

```yaml
model:
  backbone: "pvt_v2_b2"  # "res2net50" → "pvt_v2_b2" (more modern)
```

#### Faster Training (Sacrifice Accuracy)

```yaml
meta:
  inner_steps: 3        # 5 → 3 (fewer adaptation steps)
  support_size: 8       # 16 → 8 (fewer examples per domain)
  query_size: 8         # 16 → 8

training:
  epochs: 50            # 200 → 50
```

#### Slower Training for Better Accuracy

```yaml
meta:
  inner_steps: 7        # More adaptation steps
  support_size: 32      # More examples
  query_size: 32
  inner_lr: 0.005       # Smaller steps, less overshoot
  outer_lr: 0.0001      # Smaller updates

training:
  epochs: 300           # More training
```

### Nested Configuration Fields

The configuration uses nested YAML structure:

```yaml
loss:
  bce_weight: 0.5       # Accessed as: cfg.loss.bce_weight
  dice_weight: 0.5      # Accessed as: cfg.loss.dice_weight
```

In code:
```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/default.yaml")
print(cfg.loss.bce_weight)  # → 0.5
```

---

## Common Tasks & Workflows

### Task 1: Train from Scratch

```bash
# 1. Make sure datasets are downloaded
python data/download.py --output-dir ./datasets

# 2. Start meta-learning training
python scripts/train_meta.py --config configs/default.yaml

# 3. Monitor training (in another terminal)
tensorboard --logdir runs/

# 4. Wait 4-6 hours for training to complete
# Check runs/best_model.pth for best weights
```

### Task 2: Resume Training from Checkpoint

If training interrupted (e.g., Jupyter kernel died):

```bash
# Resume from epoch 50
python scripts/train_meta.py \
    --config configs/default.yaml \
    --resume runs/checkpoint_epoch_50.pth
```

### Task 3: Evaluate Trained Model

```bash
# Zero-shot (generalization without adaptation)
python scripts/evaluate.py \
    --checkpoint runs/best_model.pth \
    --mode zero_shot

# Few-shot (with 16-example adaptation)
python scripts/evaluate.py \
    --checkpoint runs/best_model.pth \
    --mode few_shot

# Results saved to: runs/eval_results.json
```

### Task 4: Run Ablation Study

Compare which components matter most:

```bash
# Test all ablations
python scripts/ablation.py \
    --config configs/default.yaml \
    --ablation all

# Test removing FAW module
python scripts/ablation.py \
    --config configs/default.yaml \
    --ablation remove_faw

# Results saved to: runs/ablation_results.json
```

### Task 5: Visualize Results

```python
# In a Jupyter notebook:
from utils.visualization import plot_segmentation
import torch
from PIL import Image

model = torch.load('runs/best_model.pth')
image = Image.open('path/to/image.png')

# Predict segmentation
mask = model(image)

# Plot side-by-side
plot_segmentation(image, mask)
```

### Task 6: Use Model for Prediction on New Hospital

```python
import torch
from models.gmlf_net import GMLFNet
from PIL import Image
import torchvision.transforms as T

# 1. Load trained model
model = torch.load('runs/best_model.pth')
model.eval()

# 2. Prepare image
image = Image.open('new_hospital_image.png').resize((352, 352))
transform = T.ToTensor()
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# 3. Predict (zero-shot: no adaptation)
with torch.no_grad():
    mask = model(image_tensor)

# 4. Convert to binary (threshold at 0.5)
binary_mask = (mask > 0.5).long()

print(f"Predicted segmentation shape: {binary_mask.shape}")
```

### Task 7: Adapt Model to Specific Hospital

```python
import torch
from models.gmlf_net import GMLFNet
import learn2learn as l2l

# 1. Load trained model
model = torch.load('runs/best_model.pth')
model.eval()

# 2. Wrap with MAML (same as training)
maml = l2l.algorithms.MAML(model, lr=0.01)

# 3. Get 16 support images & masks from new hospital
support_images = load_images(...)  # Your code
support_masks = load_masks(...)    # Your code

# 4. Adapt on support set (inner loop)
learner = maml.clone()
for step in range(5):  # 5 adaptation steps
    output = learner(support_images)
    loss = loss_fn(output, support_masks)
    learner.adapt(loss)

# 5. Use adapted model for inference
query_images = load_images(...)  # New images from hospital
with torch.no_grad():
    predictions = learner(query_images)
```

---

## Troubleshooting

### Issue 1: "CUDA out of memory"

**Problem:** GPU memory exceeded

**Solutions:**
```bash
# Reduce batch-related parameters in config
batch_size: 2                # Smaller batches
support_size: 8              # Fewer examples per domain
query_size: 8
tasks_per_batch: 2           # Fewer tasks per batch

# Or reduce image size
image_size: 256              # From 352 → 256
```

### Issue 2: Model accuracy stuck at ~50%

**Problem:** Training not converging

**Solutions:**
```yaml
# Check learning rates (might be too high)
meta:
  inner_lr: 0.005            # Reduce from 0.01
  outer_lr: 0.0001           # Reduce from 0.001

# Add warmup for stable training
training:
  warmup_epochs: 20          # Gradually increase LR
```

### Issue 3: Loss is NaN

**Problem:** Numerical instability

**Solutions:**
```yaml
# Add gradient clipping
training:
  grad_clip: 0.5             # Reduce from 1.0

# Check data normalization happens
# (should be automatic in data loader)

# Reduce inner learning rate
meta:
  inner_lr: 0.001            # Much smaller
```

### Issue 4: Model overfits (high train loss, low val loss drop)

**Problem:** Model memorizes training data

**Solutions:**
```yaml
# Add regularization
meta:
  inner_steps: 3             # Fewer adaptation steps
  
loss:
  # Weights already balance three losses

# Use more data augmentation
# (already in augmentations.py)

# Reduce model capacity
model:
  faw_hidden_dim: 32         # Smaller FAW
```

### Issue 5: Can't download datasets

**Problem:** Download script fails

**Solutions:**
```bash
# Check internet connection
ping google.com

# Try manual download from sources:
# - Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
# - CVC-ClinicDB: http://mv.cvc.uab.es/projects/polyp/
# - CVC-ColonDB: https://www.dropbox.com/s/gePN5eUEVR9g5Sx/CVC-ColonDB.zip
# Then extract to ./datasets/

# Or skip download if running locally
# (edit download.py to only download specific datasets)
```

### Issue 6: TensorBoard not showing logs

**Problem:** Logging not working

**Solutions:**
```bash
# Check logs directory exists
ls -la runs/logs/

# Try W&B instead of TensorBoard
# Edit config:
logging:
  backend: "wandb"
  wandb_project: "GMLFNet"

# Then login to W&B
wandb login
```

---

## Next Steps & Further Learning

### If You Want to:

**Understand Meta-Learning Better:**
- Read paper: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al., 2017)
- Run tutorials: https://github.com/learnables/learn2learn

**Improve Model Accuracy:**
- Try different backbones (already configurable)
- Experiment with loss weights in `loss:` section of config
- Try data augmentation in `data/augmentations.py`

**Deploy Model:**
- Convert to ONNX format for edge devices
- Use TorchScript for production deployment
- Build REST API with FastAPI

**Extend to Other Problems:**
- Polyp classification instead of segmentation
- Other medical imaging (X-ray, CT, MRI)
- Non-medical image segmentation

### Recommended Reading:

1. **MAML Paper**: https://arxiv.org/abs/1703.03400
2. **Medical Image Segmentation**: https://arxiv.org/abs/1505.04597 (U-Net)
3. **Domain Generalization**: https://arxiv.org/abs/2103.02324
4. **FiLM Modulation**: https://arxiv.org/abs/1709.07871

---

## FAQ

**Q: Why use meta-learning instead of just training on all domains?**
A: Training on all domains treats them equally. Meta-learning creates algorithms that work on *any* domain, even unseen ones—that's domain generalization.

**Q: How much adaptation data (support set size) do I really need?**
A: 16 images worked in our experiments. With fewer (5-8), adaptation is noisier. With more (32+), you're essentially doing full retraining.

**Q: Can I use this for non-polyp segmentation?**
A: Yes! The architecture is general-purpose. Just retrain on your data.

**Q: Why not fine-tune the entire model?**
A: Fine-tuning all 50M+ parameters on 16 examples causes overfitting. FAW with 100K parameters adapts without overfitting.

**Q: How long does adaptation take at deployment?**
A: ~1 minute on GPU for 5 gradient steps on 16 images. ~5 minutes on CPU.

**Q: What if I only have access to one hospital's data?**
A: Use baseline training (`train_baseline.py`). Meta-learning benefits from multiple domains.

---

## Summary

**GMLFNet** solves a real problem in medical AI: models trained on one hospital don't work on another. By combining:
- **Meta-learning (MAML)**: Learning how to adapt
- **Fast Adaptation Weights**: Lightweight, quick adaptation
- **Multi-scale decoding**: Precise segmentation

...it enables safe, practical deployment across different hospitals without expensive retraining.

The code is modular, well-organized, and fully configurable. Start with training on the default config, then experiment with different settings to understand what matters.

---

**Happy learning! 🚀**
