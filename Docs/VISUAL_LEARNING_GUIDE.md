# GMLFNet Visual Learning Guide
## Understanding Meta-Learning with Diagrams

---

## 1. The Problem: Domain Shift

### What is Domain Shift?

```
Hospital A                          Hospital B
(Training)                          (Testing - Unseen!)

[Bright, Clear Image]              [Dark, Blurry Image]
  of Polyp                           of Polyp
       ↓                                 ↓
   [Model]                          [Model]
   Good!                            Bad!
   Dice: 0.85                       Dice: 0.50
   
   ✓ Accurate                       ✗ Inaccurate
```

### Why This Happens

```
Image differences across hospitals:

Resolution    Lighting       Color        Contrast
──────────────────────────────────────────────────
Hospital A:   512×512        Bright       Warm         High
Hospital B:   640×480        Dim          Cool         Low
Hospital C:   1024×768       Mixed        Natural      Medium

Same polyp, completely different appearance!
→ Model trained on A fails on B and C
→ Need transfer learning or adaptation
```

---

## 2. Traditional Approaches vs Meta-Learning

### Approach 1: Fine-Tune on New Hospital

```
Old Model
(trained on Hospital A)
       ↓
┌──────────────────┐
│ Get 100 images   │
│ from Hospital B  │
└──────────────────┘
       ↓
   Fine-tune
   (update all 50M parameters)
       ↓
New Model
(for Hospital B)

❌ Problem 1: Need 100 images (expensive)
❌ Problem 2: Retraining takes hours
❌ Problem 3: Might overfit with only 100 images
```

### Approach 2: GMLFNet (Meta-Learning)

```
┌─────────────────────────────────────────┐
│ Meta-Train on 3 hospitals (A, B, C)    │
│ Learn how to adapt (meta-learning)      │
└─────────────────────────────────────────┘
       ↓
General Model
(knows how to generalize)
       ↓
┌─────────────────────────────────────────┐
│ Get 16 images from Hospital D (new)    │
│ Adapt using learned adaptation strategy │
│ (5 gradient steps on FAW only)          │
└─────────────────────────────────────────┘
       ↓
Adapted Model
(for Hospital D)

✓ Benefit 1: Only 16 images needed (10x fewer)
✓ Benefit 2: Adaptation takes 1 minute
✓ Benefit 3: Doesn't overfit (only 100K params changed)
```

---

## 3. Understanding MAML: Inner Loop vs Outer Loop

### The Learning Process Analogy

**Learning to play Tennis (Analogy):**

```
OUTER LOOP: "Improve as a tennis player overall"
│
├─ INNER LOOP 1: "Practice on clay court for 30 mins"
│  ├─ Hit forehands (adapt to clay)
│  ├─ Hit backhands (adapt to clay)
│  └─ Evaluate: Can I play better on clay now?
│  → YES, but only because I practiced
│
├─ INNER LOOP 2: "Practice on grass court for 30 mins"
│  ├─ Hit forehands (adapt to grass)
│  ├─ Hit backhands (adapt to grass)
│  └─ Evaluate: Can I play better on grass now?
│  → YES, but only because I practiced
│
└─ KEY INSIGHT: 
   Best overall player = one who learns quickly
   on ANY surface, not just one specific surface
```

### MAML in Image Segmentation

```
OUTER LOOP (Blue): "Improve model for any hospital"
┌──────────────────────────────────────────────┐
│                                              │
│  INNER LOOP 1 (Red): Hospital A Adaptation  │
│  ┌─────────────────────────────────────┐   │
│  │ [Clone model → Take 5 gradient steps│   │
│  │  on 16 Hospital A images → Evaluate│   │
│  │  on 16 Hospital A test images]     │   │
│  │                                     │   │
│  │  Loss_A = 0.30                      │   │
│  └─────────────────────────────────────┘   │
│                                              │
│  INNER LOOP 2 (Green): Hospital B Adapt.    │
│  ┌─────────────────────────────────────┐   │
│  │ [Clone model → Take 5 gradient steps│   │
│  │  on 16 Hospital B images → Evaluate│   │
│  │  on 16 Hospital B test images]     │   │
│  │                                     │   │
│  │  Loss_B = 0.31                      │   │
│  └─────────────────────────────────────┘   │
│                                              │
│  INNER LOOP 3 (Purple): Hospital C Adapt.   │
│  ┌─────────────────────────────────────┐   │
│  │ [Clone model → Take 5 gradient steps│   │
│  │  on 16 Hospital C images → Evaluate│   │
│  │  on 16 Hospital C test images]     │   │
│  │                                     │   │
│  │  Loss_C = 0.32                      │   │
│  └─────────────────────────────────────┘   │
│                                              │
│  Total_Loss = Loss_A + Loss_B + Loss_C     │
│            = 0.30 + 0.31 + 0.32 = 0.93    │
│                                              │
│  Update ALL model parameters to minimize   │
│  Total_Loss (so all hospitals adapt faster)│
│                                              │
│  ✓ After update: model is "meta-trained"  │
└──────────────────────────────────────────────┘
       ↓
   Repeat for 200 epochs
```

---

## 4. Fast Adaptation Weights (FAW): The Key Innovation

### Problem: Expensive Inner Loop Adaptation

```
Normal MAML:
┌────────────────────────────────────┐
│ Clone entire model (50M params)    │
│ Clone entire decoder (30M params)  │
│ Take 5 adaptation steps            │
│ → Expensive! Slow!                 │
└────────────────────────────────────┘

GMLFNet with FAW:
┌────────────────────────────────────┐
│ Clone ONLY FAW (100K params)       │
│ Other params stay on original      │
│ Take 5 adaptation steps            │
│ → 500x fewer params! Fast!         │
└────────────────────────────────────┘
```

### How FAW Works: FiLM Modulation

```
Step 1: Extract domain characteristics
────────────────────────────────────────
Feature from Encoder = [f1, f2, f3, f4]  (multi-scale)

Step 2: Pool global statistics
────────────────────────────────────────
  Global Avg Pool(f1) = [0.5, 0.3, 0.4, ...]  (128 values)
  Global Avg Pool(f2) = [0.6, 0.4, 0.5, ...]  (256 values)
  Global Avg Pool(f3) = [0.7, 0.5, 0.6, ...]  (512 values)
  Global Avg Pool(f4) = [0.8, 0.6, 0.7, ...]  (1024 values)
  
  Concatenated = [0.5, 0.3, 0.4, ..., 0.8, 0.6, 0.7, ...]
                 1920 values total (domain descriptor)

Step 3: Lightweight MLP prediction
────────────────────────────────────────
  MLP: 1920 → 64 → 32
       ↓       ↓
      [activation layers]
       ↓       ↓
       ↓      Output: [γ1, β1, γ2, β2, γ3, β3]
       └──── (6 modulation parameters, one per layer pair)

Step 4: FiLM Modulation
────────────────────────────────────────
  For decoder layer i:
  
  modulated_feature = γ_i × feature + β_i
  
  Where:
    γ_i = scale (multiply)        → Adjust intensity
    β_i = shift (add)             → Adjust brightness
  
  Visual example:
  ┌──────────────────────────────────┐
  │ Original Feature:                 │
  │ [0.2, -0.1, 0.5, 0.3]           │
  │                                   │
  │ With γ=2.0, β=0.1:               │
  │ [0.2×2+0.1, -0.1×2+0.1, ...]    │
  │ = [0.5, -0.1, 1.1, 0.7]         │
  │                                   │
  │ Effect: Scaled + shifted         │
  │ to match new hospital style      │
  └──────────────────────────────────┘
```

### Why This Strategy Works

```
┌────────────────────────────────────────────────┐
│ Encoder (50M params)                          │
│ ✗ DON'T change                               │
│ → Already learned good features               │
│ → Keep generalization ability                 │
└────────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────────┐
│ FAW (100K params)                             │
│ ✓ ADAPT THIS                                 │
│ → Lightweight modulation only                 │
│ → Can change quickly (5 steps)                │
│ → Captures domain-specific style              │
└────────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────────┐
│ Decoder (30M params)                          │
│ ✗ DON'T change                               │
│ → Already learned good upsampling             │
│ → Modulation handles style differences        │
└────────────────────────────────────────────────┘
```

---

## 5. Complete Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ PRE-TRAINING                                                    │
│ Load pretrained backbones (ImageNet weights on encoder)        │
│ (This saves us from training from scratch)                      │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ EPOCH 1/200                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ EPISODE 1/100 (sample 3 random hospitals)                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ INNER LOOP 1: Hospital Kvasir                           │   │
│ │ • Load 16 support images + 16 query images             │   │
│ │ • Forward pass on support: output = model(support)     │   │
│ │ • Compute loss: loss_s = criterion(output, target)    │   │
│ │ • Backward: loss_s.backward()                          │   │
│ │ • Update FAW only: optimizer.step()                    │   │
│ │ • Repeat 5 times (inner_steps)                         │   │
│ │ • Forward on query: output_q = model(query)           │   │
│ │ • Compute loss: loss_q = criterion(output_q, target_q)│   │
│ │ • Accumulate: total_loss += loss_q                     │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ INNER LOOP 2: Hospital CVC-ClinicDB                            │
│ [Same process as above]                                         │
│                                                                 │
│ INNER LOOP 3: Hospital CVC-ColonDB                             │
│ [Same process as above]                                         │
│                                                                 │
│ OUTER LOOP:                                                     │
│ • total_loss = loss_from_kvasir + loss_from_cvc_clinic +      │
│                loss_from_cvc_colon                             │
│ • Backward: total_loss.backward()    (through all params)     │
│ • Outer optimizer step: outer_optimizer.step()                │
│ • Scheduler step: scheduler.step()                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        ↓
    (repeat 100 episodes per epoch)
        ↓
    (repeat 200 epochs)
        ↓
┌─────────────────────────────────────────────────────────────────┐
│ MODEL SAVED                                                     │
│ runs/best_model.pth (best validation dice score)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Inference Time: Two Strategies

### Strategy 1: Zero-Shot (No Adaptation)

```
┌─────────────────────────────┐
│ Load pretrained model       │
│ (runs/best_model.pth)       │
└─────────────────────────────┘
        ↓
   (deployed to new hospital)
        ↓
┌─────────────────────────────┐
│ Test image from Hospital D  │
│ (never seen during training)│
└─────────────────────────────┘
        ↓
   model(image)  ← Direct inference
        ↓
┌─────────────────────────────┐
│ Segmentation mask           │
│ Dice Score: 0.76            │
│ mIoU: 0.65                  │
└─────────────────────────────┘

Time: 0.1 seconds per image
Accuracy: Good (60-80% typical)
Effort: None (just deploy)
```

### Strategy 2: Few-Shot (With Adaptation)

```
┌─────────────────────────────┐
│ Load pretrained model       │
│ (runs/best_model.pth)       │
└─────────────────────────────┘
        ↓
   (deployed to new hospital)
        ↓
┌─────────────────────────────┐
│ Get 16 support images       │
│ from new hospital's dataset │
└─────────────────────────────┘
        ↓
  Adaptation Loop (5 steps)
  ┌─────────────────────────┐
  │ Forward on support:     │
  │ output = model(support) │
  │ loss = criterion(...)   │
  │ loss.backward()         │
  │ optimizer.step()        │
  │ (update FAW only)       │
  └─────────────────────────┘
        ↓
  (repeat 5 times)
        ↓
┌─────────────────────────────┐
│ Test images from Hospital D │
│ (same hospital as support)  │
└─────────────────────────────┘
        ↓
   adapted_model(test_image)  ← Inference with adapted params
        ↓
┌─────────────────────────────┐
│ Segmentation mask           │
│ Dice Score: 0.84            │
│ mIoU: 0.73                  │
└─────────────────────────────┘

Time: 1 minute (adaptation) + 0.1 sec/image
Accuracy: Excellent (75-90% typical)
Improvement: +8% Dice score vs zero-shot
Effort: Collect 16 labeled examples
```

---

## 7. Key Parameters Explained Visually

### Inner Learning Rate vs Outer Learning Rate

```
Inner Learning Rate (inner_lr: 0.01)
┌────────────────────────────────────────────┐
│ Controls how quickly FAW adapts            │
│                                             │
│ High (0.1):  Large steps → Fast adapt      │
│ └─ Risk: Overshoot, unstable              │
│                                             │
│ Medium (0.01): Balanced (default)          │
│ └─ Good for most cases                    │
│                                             │
│ Low (0.001): Small steps → Stable         │
│ └─ Risk: Too slow to adapt                │
└────────────────────────────────────────────┘

Outer Learning Rate (outer_lr: 0.001)
┌────────────────────────────────────────────┐
│ Controls how quickly we improve meta-learning│
│                                             │
│ High (0.01):  Large updates → Fast progress│
│ └─ Risk: Unstable, diverge               │
│                                             │
│ Medium (0.001): Balanced (default)         │
│ └─ Good for convergence                   │
│                                             │
│ Low (0.0001): Small updates → Very stable │
│ └─ Risk: Takes very long to train        │
└────────────────────────────────────────────┘
```

### Inner Steps vs Training Curve

```
inner_steps: 3
┌──────────────────────────────┐
│ Training Speed: Fast         │
│ Memory: Low                  │
│ Adaptation Quality: OK       │
│                              │
│ Loss: ████   (drops slowly)  │
│ Training time: 2 hours       │
│ Accuracy: 65-70%             │
└──────────────────────────────┘

inner_steps: 5 (default)
┌──────────────────────────────┐
│ Training Speed: Medium       │
│ Memory: Medium               │
│ Adaptation Quality: Good     │
│                              │
│ Loss: ██████ (drops well)    │
│ Training time: 4-5 hours     │
│ Accuracy: 75-80%             │
└──────────────────────────────┘

inner_steps: 10
┌──────────────────────────────┐
│ Training Speed: Slow         │
│ Memory: High                 │
│ Adaptation Quality: Excellent│
│                              │
│ Loss: ████████ (drops fast) │
│ Training time: 8 hours       │
│ Accuracy: 78-82%             │
└──────────────────────────────┘
```

---

## 8. Loss Function Visualization

### Three Types of Loss

```
1. BINARY CROSS-ENTROPY (BCE)
   ┌─────────────────────────────────┐
   │ Pixel-level classification loss │
   │                                 │
   │ Prediction: 0.7 (70% polyp)    │
   │ Target:     1.0 (100% polyp)   │
   │                                 │
   │ BCE Loss = -[1×log(0.7) +      │
   │            (1-1)×log(1-0.7)]  │
   │          ≈ 0.36                │
   │                                 │
   │ Use: Handles probability calibration
   └─────────────────────────────────┘

2. DICE LOSS
   ┌─────────────────────────────────┐
   │ Region overlap loss             │
   │                                 │
   │ Predicted region: ████ (100k px)│
   │ Actual region:    ██  (50k px) │
   │ Overlap:          ██  (40k px) │
   │                                 │
   │ Dice = 2×40/(100+50) ≈ 0.53    │
   │ Dice Loss = 1 - 0.53 = 0.47    │
   │                                 │
   │ Use: Handles class imbalance     │
   │      (few polyp pixels vs many background)
   └─────────────────────────────────┘

3. STRUCTURE LOSS (Boundary)
   ┌─────────────────────────────────┐
   │ Boundary preservation            │
   │                                 │
   │ Predicted edge:    ▓▒░         │
   │ Actual edge:       ░░░         │
   │                                 │
   │ Difference: sharp vs blurry     │
   │ Structure Loss: measures edge difference
   │                                 │
   │ Use: Keep boundaries sharp      │
   │      (important for precision)  │
   └─────────────────────────────────┘

Total Loss = 0.5 × BCE + 0.5 × Dice + 0.2 × Structure
           = 0.5×0.36 + 0.5×0.47 + 0.2×0.15
           ≈ 0.41
```

---

## 9. Evaluation Metrics

### Dice Score Visualized

```
Perfect (Dice = 1.0)
┌─────────────────┐
│ Prediction: ███ │
│ Ground Truth: ███
│ Overlap: ███    │
└─────────────────┘

Good (Dice = 0.80)
┌─────────────────┐
│ Prediction: ████ │
│ Ground Truth: ███
│ Overlap: ██     │
└─────────────────┘

Okay (Dice = 0.60)
┌─────────────────┐
│ Prediction: █████ │
│ Ground Truth: ███
│ Overlap: ██     │
└─────────────────┘

Poor (Dice = 0.30)
┌─────────────────┐
│ Prediction: ██ │
│ Ground Truth: ███
│ Overlap: █      │
└─────────────────┘

Formula: Dice = 2×(TP) / (2×TP + FP + FN)
TP = True Positives (correct polyp detection)
FP = False Positives (detected non-polyp as polyp)
FN = False Negatives (missed polyp)
```

### mIoU (Intersection over Union) Visualized

```
Perfect (mIoU = 1.0)
┌─────────────────────────────┐
│ Prediction:     ███         │
│ Ground Truth:   ███         │
│ Union:          ███         │
│ Intersection:   ███         │
│ mIoU = 3/3 = 1.0           │
└─────────────────────────────┘

Good (mIoU = 0.70)
┌─────────────────────────────┐
│ Prediction:     ████        │
│ Ground Truth:   ███         │
│ Union:          █████       │
│ Intersection:   ██          │
│ mIoU = 2/5 = 0.70          │
└─────────────────────────────┘

Formula: mIoU = (TP) / (TP + FP + FN)
Stricter than Dice (requires exact boundaries)
```

---

## 10. Comparison: Different Configurations

### Effect of Configuration Changes

```
Config 1: Small (Fast)
├─ backbone: res2net50
├─ image_size: 256
├─ inner_steps: 3
├─ support_size: 8
├─ epochs: 50
├─ Training time: 1 hour
└─ Accuracy: ~70% Dice

Config 2: Medium (Default, Recommended)
├─ backbone: res2net50
├─ image_size: 352
├─ inner_steps: 5
├─ support_size: 16
├─ epochs: 200
├─ Training time: 4-5 hours
└─ Accuracy: ~78% Dice ← PICK THIS

Config 3: Large (Slow, Best)
├─ backbone: pvt_v2_b2
├─ image_size: 512
├─ inner_steps: 7
├─ support_size: 32
├─ epochs: 300
├─ Training time: 12+ hours
└─ Accuracy: ~82% Dice

Config 4: Baseline (No Meta-Learning)
├─ Just standard supervised learning
├─ Training time: 3 hours
├─ Zero-shot accuracy: ~65% Dice
└─ Few-shot accuracy: Same (~65%)
```

---

## 11. Decision Tree: What to Do When

```
START
  │
  ├─ "I want to understand the project"
  │  └─ Read: QUICK_REFERENCE.md first, then COMPLETE_DOCUMENTATION.md
  │
  ├─ "I want to train the model"
  │  └─ Run: python scripts/train_meta.py --config configs/default.yaml
  │
  ├─ "Training is too slow"
  │  ├─ Option 1: Reduce image_size (352 → 256)
  │  ├─ Option 2: Reduce inner_steps (5 → 3)
  │  ├─ Option 3: Reduce epochs (200 → 50)
  │  └─ Option 4: Use faster backbone (res2net50 is already fast)
  │
  ├─ "Model accuracy is bad"
  │  ├─ Check: Is training loss decreasing? (If no, learning rate issue)
  │  ├─ Action: Increase epochs (200 → 300)
  │  ├─ Action: Use better backbone (res2net50 → pvt_v2_b2)
  │  └─ Action: Increase inner_steps (5 → 7)
  │
  ├─ "CUDA out of memory"
  │  ├─ Action: Reduce support_size (16 → 8)
  │  ├─ Action: Reduce query_size (16 → 8)
  │  └─ Action: Reduce image_size (352 → 256)
  │
  ├─ "I want to test the model"
  │  ├─ Zero-shot: python scripts/evaluate.py --checkpoint ... --mode zero_shot
  │  └─ Few-shot:  python scripts/evaluate.py --checkpoint ... --mode few_shot
  │
  ├─ "I want to understand which parts matter"
  │  └─ Run: python scripts/ablation.py --ablation all
  │
  └─ "I want to deploy to new hospital"
     ├─ If only need quick predictions: Use zero-shot (no adaptation)
     └─ If can collect 16 examples: Use few-shot (with 1-min adaptation)
```

---

## Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **Domain Shift** | Images at different hospitals look different → models fail |
| **Meta-Learning** | Don't learn to do a task, learn how to adapt to any task |
| **MAML** | Inner loop adapts to specific domain, outer loop improves for all domains |
| **FAW** | Adapt with 100K params instead of 50M → fast & safe |
| **FiLM** | Simple scaling & shifting → effective domain adaptation |
| **Zero-Shot** | Test without adaptation → measures generalization |
| **Few-Shot** | Adapt with 16 examples → measures quick adaptability |

---

**Ready to dive deeper? Start with QUICK_REFERENCE.md, then COMPLETE_DOCUMENTATION.md!**
