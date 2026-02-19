# GMLFNet Quick Reference Guide
## Beginner's Cheat Sheet

---

## 🚀 Quick Start (5 Minutes)

### Installation
```bash
pip install -r requirements.txt
python data/download.py --output-dir ./datasets
```

### Train Model
```bash
python scripts/train_meta.py --config configs/default.yaml
```

### Test Model
```bash
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode zero_shot
```

---

## 📚 Key Concepts at a Glance

| Concept | Simple Explanation | Why It Matters |
|---------|-------------------|----------------|
| **Polyp Segmentation** | Finding which pixels are polyp vs background | Precision needed for surgery |
| **Domain Shift** | Images look different at different hospitals | Model breaks when deployed |
| **Meta-Learning** | Learning how to learn, not just solving one task | Works on unseen hospitals |
| **MAML** | Optimize weights so they adapt quickly to new tasks | Standard meta-learning algorithm |
| **FAW (Fast Adaptation Weights)** | Lightweight modulation parameters (100K not 50M) | Quick & safe adaptation |
| **FiLM Modulation** | Scale & shift features: `gamma * feature + beta` | Simple but effective adaptation |

---

## 🏗️ Architecture in One Picture

```
Input Image (352×352)
       ↓
┌──────────────────────┐
│  ENCODER (50M params)│ ← Pretrained, stays frozen mostly
│  Extract 4 scales    │
└──────────────────────┘
       ↓ [F1, F2, F3, F4]
┌──────────────────────┐
│  FAW (100K params)   │ ← ADAPTS TO NEW HOSPITAL
│  Generate gamma/beta │
└──────────────────────┘
       ↓ [modulated features]
┌──────────────────────┐
│  DECODER (30M params)│ ← Pretrained, stays mostly frozen
│  Reconstruct mask    │
└──────────────────────┘
       ↓
Output Mask (352×352, values 0-1)
```

---

## 🎓 Training vs Evaluation

### During Training (Meta-Learning)
```
┌─ OUTER LOOP: Update all parameters ──────┐
│                                           │
│  ┌─ INNER LOOP: Hospital A ────────────┐ │
│  │ • Copy model                        │ │
│  │ • Take 5 gradient steps on 16 imgs  │ │
│  │ • Test on 16 query imgs             │ │
│  │ Compute loss_A                      │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  ┌─ INNER LOOP: Hospital B ────────────┐ │
│  │ • Copy model                        │ │
│  │ • Take 5 gradient steps on 16 imgs  │ │
│  │ • Test on 16 query imgs             │ │
│  │ Compute loss_B                      │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  ┌─ INNER LOOP: Hospital C ────────────┐ │
│  │ • Copy model                        │ │
│  │ • Take 5 gradient steps on 16 imgs  │ │
│  │ • Test on 16 query imgs             │ │
│  │ Compute loss_C                      │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  Backprop: loss_A + loss_B + loss_C      │
│  Update all parameters                   │
└───────────────────────────────────────────┘
```

### After Training: Deployment (Two Options)

**Option 1: Zero-Shot** (No Adaptation)
```
[Pretrained Model] → [Test Image] → [Segmentation]

Speed: Fast (~0.1 sec per image)
Accuracy: Good (model already generalized)
```

**Option 2: Few-Shot** (Adapt First)
```
[Pretrained Model] → Adapt on 16 examples → [Test Image] → [Segmentation]

Speed: Slow (~1 min to adapt + 0.1 sec per image)
Accuracy: Better (adapted to hospital's style)
```

---

## 📁 File Purpose Reference

```
configs/
  default.yaml              → MODIFY THIS to change training settings

data/
  datasets.py               → How images are loaded
  meta_sampler.py           → Creates training episodes
  download.py               → Get the datasets

models/
  gmlf_net.py               → Main model (encoder + FAW + decoder)
  fast_adapt_weights.py     → THE KEY INNOVATION
  backbone.py               → Res2Net or ViT encoder
  decoder.py                → Upsampling decoder
  losses.py                 → What we optimize

trainers/
  meta_trainer.py           → MAML training loop
  baseline_trainer.py       → Normal training
  evaluator.py              → Testing

scripts/
  train_meta.py             → RUN THIS to train
  evaluate.py               → RUN THIS to test
  ablation.py               → RUN THIS to compare versions

utils/
  metrics.py                → Dice, IoU, MAE
  visualization.py          → Plot images & masks
  logging_utils.py          → Save results
```

---

## 💻 Command Reference

### Training Commands

```bash
# Train (default config, 200 epochs, 4-6 hours)
python scripts/train_meta.py --config configs/default.yaml

# Train faster (50 epochs, proof-of-concept)
# (Edit config: epochs: 50)
python scripts/train_meta.py --config configs/custom.yaml

# Resume from checkpoint (if interrupted)
python scripts/train_meta.py --config configs/default.yaml \
    --resume runs/checkpoint_epoch_50.pth

# Train baseline (no meta-learning, for comparison)
python scripts/train_baseline.py --config configs/default.yaml

# Monitor training (separate terminal)
tensorboard --logdir runs/
# Then open: http://localhost:6006
```

### Evaluation Commands

```bash
# Zero-shot (test without adaptation)
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode zero_shot

# Few-shot (test with 16-example adaptation)
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode few_shot

# Both modes
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode both
```

### Ablation Commands

```bash
# Test all ablations
python scripts/ablation.py --config configs/default.yaml --ablation all

# Test effect of inner loop steps
python scripts/ablation.py --config configs/default.yaml --ablation inner_steps

# Test different backbones
python scripts/ablation.py --config configs/default.yaml --ablation backbone
```

---

## ⚙️ Configuration Quick Reference

### Data Settings
```yaml
data:
  image_size: 352          # Increase → more memory, better quality
  support_size: 16         # More → better adaptation, slower training
  query_size: 16           # More → better evaluation, slower training
```

### Model Settings
```yaml
model:
  backbone: "res2net50"    # Fast, reliable
  # backbone: "pvt_v2_b2"  # Modern, better, slower
  faw_hidden_dim: 64       # Smaller → faster, less capacity
```

### Meta-Learning Settings
```yaml
meta:
  inner_lr: 0.01           # Smaller → more stable, slower adaptation
  inner_steps: 5           # More → better adaptation, more compute
  outer_lr: 0.001          # Smaller → more stable but slower
```

### Training Settings
```yaml
training:
  epochs: 200              # More → better, longer training
  grad_clip: 1.0           # Prevent exploding gradients
  scheduler: "cosine"      # Learning rate schedule
```

### Loss Settings
```yaml
loss:
  bce_weight: 0.5          # Binary classification loss weight
  dice_weight: 0.5         # Region overlap loss weight
  structure_weight: 0.2    # Boundary preservation weight
```

---

## 📊 Understanding the Output

### Training Metrics (What They Mean)

```
Epoch 1/200
  Train Loss: 0.45
  ├─ BCE Loss: 0.25 (binary classification error)
  ├─ Dice Loss: 0.15 (overlap error)
  └─ Structure Loss: 0.05 (boundary error)
  
  Val Loss: 0.42
  Dice Score: 0.82 (↑ is better, 1.0 is perfect)
  mIoU: 0.71 (↑ is better, 1.0 is perfect)
```

### Evaluation Metrics (After Training)

```
Zero-Shot Results (no adaptation):
  Dataset               Dice Score    mIoU
  ─────────────────────────────────────────
  ETIS-LaribPolypDB      0.78         0.65
  CVC-300                0.75         0.62
  Average                0.765        0.635

Few-Shot Results (with adaptation):
  Dataset               Dice Score    mIoU
  ─────────────────────────────────────────
  ETIS-LaribPolypDB      0.85         0.75
  CVC-300                0.82         0.71
  Average                0.835        0.73
```

**Interpretation:**
- Dice: 0.85 means 85% overlap between predicted and actual polygon
- mIoU: 0.75 means 75% intersection over union
- Higher is better for both (1.0 = perfect)

---

## 🔍 Common Problems & Quick Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA out of memory | Batch too large | Reduce `support_size`, `query_size`, `tasks_per_batch` |
| Loss = NaN | Unstable gradients | Reduce `inner_lr` or increase `grad_clip` |
| Accuracy stuck at 50% | Not converging | Increase `epochs` or decrease `outer_lr` |
| Model overfits | Too much capacity | Reduce `faw_hidden_dim`, increase `inner_steps` |
| Training very slow | Too many parameters | Reduce `image_size` or `backbone` capacity |
| Accuracy still low | Model capacity too small | Increase `faw_hidden_dim`, use "pvt_v2_b2" |

---

## 🎯 Typical Workflow

### Step 1: Setup (5 min)
```bash
pip install -r requirements.txt
python data/download.py --output-dir ./datasets
```

### Step 2: Quick Test (10 min)
```bash
# Edit configs/default.yaml: epochs: 2
python scripts/train_meta.py --config configs/default.yaml
# Check it runs without crashes
```

### Step 3: Real Training (4-6 hours)
```bash
# Edit configs/default.yaml: epochs: 200
python scripts/train_meta.py --config configs/default.yaml
# Monitor in another terminal: tensorboard --logdir runs/
```

### Step 4: Evaluation (5 min)
```bash
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode both
# Check results in runs/eval_results.json
```

### Step 5: Ablations (1 hour)
```bash
python scripts/ablation.py --config configs/default.yaml --ablation all
# Compare which components matter
```

---

## 📈 Expected Results

After full training (200 epochs):

| Metric | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| Dice (avg) | 0.76-0.80 | 0.82-0.86 |
| mIoU (avg) | 0.63-0.68 | 0.72-0.77 |
| MAE (avg) | 0.08-0.12 | 0.06-0.09 |

**What these mean:**
- Zero-shot: Model generalizes to new hospital without adaptation
- Few-shot: Model adapts to new hospital with 16 examples
- Few-shot should be 5-10% better than zero-shot (the meta-learning benefit)

---

## 🧠 Mental Model Summary

Think of it like learning to cook:

1. **Traditional Learning (Like baseline training)**
   - Learn to make 1 specific dish perfectly
   - Fails if ingredients slightly different

2. **Meta-Learning (Like GMLFNet)**
   - Learn *how to learn* to make dishes
   - Given new ingredients, adapt in 5 steps
   - Works even if you've never seen these ingredients

3. **FAW Module**
   - Instead of learning new recipes (heavy params)
   - Just learn to adjust seasonings (light params)
   - Quick, safe, effective

---

## 💡 Pro Tips

1. **Start with pre-trained backbones** (enabled by default)
   - Much faster to train than training from scratch

2. **Monitor GPU memory** if you have large models
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Use TensorBoard for visualization**
   ```bash
   tensorboard --logdir runs/ --port 6006
   ```

4. **Save configs for reproducibility**
   ```bash
   cp configs/default.yaml configs/experiment_v1.yaml
   # Modify experiment_v1.yaml
   python scripts/train_meta.py --config configs/experiment_v1.yaml
   ```

5. **Create ablation study spreadsheet**
   - Compare Dice/mIoU across different configs
   - Track which settings matter most

---

## 📞 Need Help?

1. Check error message in terminal
2. Look at similar issue in [Troubleshooting](COMPLETE_DOCUMENTATION.md#troubleshooting) section
3. Check config is valid YAML (proper indentation)
4. Try reducing complexity (smaller image, fewer epochs)

---

## 📚 Learning Path

Recommended order to understand the code:

1. Read this file (overview)
2. Read `COMPLETE_DOCUMENTATION.md` (detailed)
3. Look at `configs/default.yaml` (understand settings)
4. Read `trainers/meta_trainer.py` (understand training)
5. Read `models/gmlf_net.py` (understand model)
6. Read `models/fast_adapt_weights.py` (understand key innovation)
7. Experiment with configs and training

---

**Happy training! 🎉**
