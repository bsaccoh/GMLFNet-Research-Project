# GMLFNet Glossary & Terminology Guide
## Understand the Jargon

---

## Machine Learning Fundamentals

### Deep Learning
**Definition:** Machine learning using artificial neural networks with many layers.

**Simple Explanation:** The computer learns patterns by passing data through layers of mathematical operations, getting better with practice.

**In Context:** We use deep learning (PyTorch) to process images and make predictions.

**Example:**
```
Input Image → Layer 1 → Layer 2 → ... → Layer 50 → Output Prediction
```

---

### Neural Network / Model
**Definition:** A computational structure inspired by biological neurons.

**Simple Explanation:** Mathematical functions with adjustable "weights" that learn from data.

**In Context:** GMLFNet is a neural network with ~80M parameters (adjustable weights).

**Visualization:**
```
Input ──[weight*input + bias]──> Output
        [activation function]
```

---

### Parameter / Weight
**Definition:** A number in the model that gets adjusted during training.

**Simple Explanation:** The "knobs" that the model twists to improve predictions.

**In Context:** 
- Encoder: 50M parameters
- FAW: 100K parameters ← We focus on adapting these
- Decoder: 30M parameters

**Analogy:** Like adjusting recipe ingredients to improve taste.

---

### Training / Optimization
**Definition:** The process of adjusting parameters to minimize error.

**Simple Explanation:** The model learns by trying, making mistakes, and correcting itself.

**In Context:** We train for 200 epochs (200 passes through the data).

**Formula:**
```
Loss = how wrong predictions are
Goal: Minimize loss by adjusting weights
```

---

### Loss Function / Objective
**Definition:** A mathematical function measuring prediction error.

**Simple Explanation:** A score of "how bad the model is" - lower is better.

**In Context:** We use combined loss: BCE + Dice + Structure

**Example:**
```
Loss = 0.5 (bad)   ← Model makes large mistakes
Loss = 0.1 (good)  ← Model makes small mistakes
Loss = 0.0 (perfect) ← Model is perfect
```

---

### Epoch
**Definition:** One complete pass through the entire training dataset.

**Simple Explanation:** Going through all your examples once.

**In Context:** Training for 200 epochs = seeing all data 200 times

**Timeline:**
```
Epoch 1: See all training samples → Update weights
Epoch 2: See all training samples → Update weights
...
Epoch 200: See all training samples → Update weights
```

---

## Medical Imaging

### Polyp
**Definition:** Abnormal tissue growth in the colon/intestines.

**Simple Explanation:** A bump or growth doctors need to remove surgically.

**In Context:** We detect and segment (outline) polyps in endoscopy images.

**Why It Matters:** Early detection and removal prevents cancer.

---

### Segmentation / Semantic Segmentation
**Definition:** Labeling each pixel as one of several classes.

**Simple Explanation:** Coloring pixels: "this is polyp" (1) or "this is background" (0).

**In Context:** We output a binary mask (1=polyp, 0=background).

**Visual:**
```
Input Image:    ____[O]____      Output Mask:   ____[1]____
                |_________|                     |_________|
                                                 1=polyp
                                                 0=background
```

---

### Classification
**Definition:** Assigning one label to entire input (is it a cat or dog?).

**Simple Explanation:** One answer for whole image.

**Different from:** Segmentation (answer for each pixel)

**Example:**
```
Classification: "This image contains a polyp" (yes/no)
Segmentation: "These pixels are polyp" (pixel-by-pixel)
```

---

### Endoscopy
**Definition:** Medical procedure using a camera to view inside body.

**Simple Explanation:** Doctor inserts tube with camera to see inside intestines.

**In Context:** Endoscopy produces images we analyze to detect polyps.

---

### Biopsy / Annotation
**Definition:** Manual examination and labeling by human expert.

**Simple Explanation:** Doctor looks at image and manually marks polyps.

**In Context:** Annotated data (images + masks) needed to train models.

---

## Domain & Transfer Learning

### Domain / Distribution
**Definition:** The characteristics of data (style, appearance, etc).

**Simple Explanation:** How the images look.

**In Context:** Different hospitals = different domains
- Hospital A: Bright, clear images
- Hospital B: Dark, blurry images
- Same polyps, different "domains"

---

### Domain Shift / Distribution Shift
**Definition:** When training and test data look different.

**Simple Explanation:** Model learned to detect bright polyps, tested on dark ones.

**In Context:** Main problem GMLFNet solves.

**Example:**
```
Train on Hospital A (bright images) → Good accuracy
Test on Hospital B (dark images) → Bad accuracy
Why? Domain shift!
```

---

### Domain Generalization
**Definition:** Creating models that work on data from unknown domains.

**Simple Explanation:** Train on some variations, test on completely new variations.

**In Context:** Train on 3 hospitals, test on 2 new hospitals never seen before.

**Goal:** Transfer ability to adapt to new hospitals.

---

### Transfer Learning
**Definition:** Using knowledge learned on one task to help another task.

**Simple Explanation:** Learn from one problem, apply to similar problem.

**In Context:** Start with weights from ImageNet, fine-tune on medical images.

**Benefits:**
- Don't need to train from scratch
- Transfer general visual features
- Faster convergence

---

### Fine-Tuning
**Definition:** Adjusting pretrained weights for new task.

**Simple Explanation:** Start with weights someone else trained, tweak for your task.

**In Context:** Backbone pretrained on ImageNet, we fine-tune for polyp detection.

**Process:**
```
Pretrained Weights (ImageNet)
         ↓
    Fine-tune
         ↓
    Task-Specific Weights
```

---

## Meta-Learning Concepts

### Meta-Learning / Learning to Learn
**Definition:** Learning algorithms that improve the learning algorithm itself.

**Simple Explanation:** Don't learn to solve one thing, learn how to learn ANY thing.

**Analogy:** 
- Normal: Learn to play chess
- Meta: Learn how to learn any game quickly

**In Context:** MAML teaches model how to adapt quickly.

---

### MAML (Model-Agnostic Meta-Learning)
**Definition:** Meta-learning algorithm with inner and outer loops.

**Simple Explanation:** Two-level learning: inner loop adapts, outer loop improves adaptation.

**In Context:** Core algorithm for training GMLFNet.

**Paper:** "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., 2017)

---

### Inner Loop / Inner Optimization
**Definition:** Fast adaptation to specific domain/task.

**Simple Explanation:** "Quick practice" on one hospital's images.

**In Training Context:**
- Take 5 gradient steps on support set from Hospital A
- Evaluate on query set from Hospital A
- Measure: Can we adapt quickly?

**In Deployment Context:**
- Take 5 gradient steps on 16 examples from new hospital
- Now model is specialized for that hospital

---

### Outer Loop / Outer Optimization
**Definition:** Learning that improves future adaptation.

**Simple Explanation:** "Meta-learning" - improving the learning algorithm itself.

**In Training Context:**
- Run inner loop on 3 hospitals
- Accumulate losses from all 3
- Backprop to improve all parameters
- Question: Which weights enable fast adaptation?

**Result:** Model learns features easy to adapt.

---

### Task / Episode
**Definition:** One training example in meta-learning (entire dataset = one task).

**Simple Explanation:** A small training problem: support set (train) + query set (test).

**In Context:**
```
Episode 1:
  Support: 16 images from Hospital A (train)
  Query: 16 images from Hospital A (test)

Episode 2:
  Support: 16 images from Hospital B (train)
  Query: 16 images from Hospital B (test)
```

---

### Support Set
**Definition:** Training data for inner loop (few examples).

**Simple Explanation:** Small dataset used for quick adaptation.

**In Context:** 16 images per hospital for adaptation.

**Characteristic:** Few-shot (small, maybe 5-32 examples).

---

### Query Set
**Definition:** Test data for inner loop (evaluate after adaptation).

**Simple Explanation:** Data to test if inner loop adaptation worked.

**In Context:** 16 images per hospital for evaluation.

**Process:**
```
Support set → Inner loop adapts → Query set → Is adaptation good?
```

---

## Fast Adaptation Weights (FAW)

### Fast Adaptation Weights (FAW)
**Definition:** Lightweight modulation parameters that adapt quickly.

**Simple Explanation:** Small extra neural network (100K params) that learns domain-specific adjustments.

**Why "Fast"?** Only update small amount in inner loop.

**Why "Adaptation"?** Adapts features for new domains.

**Why "Weights"?** They are learnable parameters.

---

### FiLM Modulation
**Definition:** Feature-wise Linear Modulation - scale and shift features.

**Simple Explanation:** Multiply features by gamma, add beta.

```
Output = gamma × feature + beta

Where:
  gamma = scaling factor (0.5 = half intensity)
  beta = shifting factor (0.1 = add 0.1 everywhere)
```

**Example:**
```
Original feature:  [1.0, 2.0, 3.0, 4.0]
Gamma = 2.0:       [2.0, 4.0, 6.0, 8.0]  (doubled)
Beta = 0.5:        [2.5, 4.5, 6.5, 8.5]  (shifted up)
```

**Use Case:** Adapt to new domain by adjusting feature magnitudes.

---

### Modulation Parameter
**Definition:** The gamma and beta values in FiLM.

**Simple Explanation:** Two numbers controlling how much to scale and shift.

**Count:** 2 per decoder layer × 3 decoder layers = 6 total per forward pass

---

### Channel-wise Modulation
**Definition:** Different scale/shift for each feature channel.

**Simple Explanation:** Each color/feature gets its own adaptation.

**In Context:** FAW produces per-channel gamma/beta.

```
Channel 1: gamma=2.0, beta=0.1
Channel 2: gamma=1.5, beta=-0.2
Channel 3: gamma=0.8, beta=0.3
...
```

---

## Architecture Components

### Encoder / Backbone
**Definition:** Part that extracts features from images.

**Simple Explanation:** Processes input image, identifies important patterns.

**In Context:** Res2Net-50 or PVTv2-B2, outputs 4-level feature pyramid.

**Frozen During:** Mostly frozen in inner loop (pretrained ImageNet weights).

---

### Feature Pyramid / Multi-Scale Features
**Definition:** Representations at different scales/resolutions.

**Simple Explanation:** Same image at different zoom levels.

**Pyramid Levels:**
```
Level 1 (fine detail):   Full resolution, 256 channels
Level 2:                 1/2 resolution, 512 channels
Level 3:                 1/4 resolution, 1024 channels
Level 4 (global context): 1/8 resolution, 2048 channels
```

**Why Multi-Scale?** Small details need fine resolution, large patterns need coarse view.

---

### Decoder
**Definition:** Part that reconstructs output from features.

**Simple Explanation:** Takes features, upsamples back to image size, produces segmentation.

**In Context:** RFB modules + Reverse Attention, progressively restores objects.

**Process:**
```
Small feature map (28×28) → Upsample → 56×56 → Add more features → 112×112 → ... → 352×352
```

---

### Attention Mechanism / Attention Module
**Definition:** Neural network module that learns to focus on important parts.

**Simple Explanation:** Asks "which features matter for this task?"

**In Context:** Reverse Attention in decoder learns where polyps likely are.

**Analogy:** Like paying more attention to suspicios areas.

---

### RFB (Receptive Field Block)
**Definition:** Module that expands receptive field (views larger context).

**Simple Explanation:** Allows model to see larger image regions to understand context.

**In Context:** Used in decoder to combine information from distant regions.

---

## Evaluation Metrics

### Dice Score / Dice Coefficient
**Definition:** Overlap between predicted and actual segmentation.

**Formula:**
```
Dice = 2 × (Intersection) / (Predicted + Actual)
```

**Range:** 0 to 1
- 0.0 = No overlap (terrible)
- 0.5 = 50% overlap (okay)
- 1.0 = Perfect overlap (excellent)

**Biased Toward:** Large objects

**In Context:** Primary metric for polyp segmentation.

---

### IoU / Jaccard Index
**Definition:** Intersection over Union - strict version of overlap.

**Formula:**
```
IoU = (Intersection) / (Union)
```

**Range:** 0 to 1
- 0.0 = No overlap
- 1.0 = Perfect overlap

**Stricter Than:** Dice (requires exact boundaries).

**In Context:** Complementary metric to Dice.

---

### mIoU / Mean Intersection over Union
**Definition:** IoU averaged over multiple image / datasets.

**Use:** When you have multiple images/datasets, average their IoU values.

---

### MAE (Mean Absolute Error)
**Definition:** Average pixel-level difference.

**Formula:**
```
MAE = average of |predicted - actual|
```

**Range:** 0 to 1
- 0.0 = Perfect prediction
- 1.0 = Completely wrong

**In Context:** Secondary metric, complements Dice.

---

### Sensitivity / True Positive Rate (TPR)
**Definition:** Of actual polyps, how many did we detect?

**Formula:**
```
Sensitivity = TP / (TP + FN)
```

**Use:** Medical context: missing polyps is dangerous.

**Goal:** High sensitivity (detect all polyps).

---

### Specificity / True Negative Rate (TNR)
**Definition:** Of non-polyp regions, how many correctly identified?

**Formula:**
```
Specificity = TN / (TN + FP)
```

**Use:** Avoid false alarms.

**Goal:** High specificity (avoid false positives).

---

### Accuracy / Pixel Accuracy
**Definition:** Percentage of pixels labeled correctly.

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Problem:** Biased by class imbalance (many background pixels).

**Not Used For:** Segmentation (Dice preferred).

---

## Training Concepts

### Gradient Descent
**Definition:** Algorithm that adjusts weights in direction that reduces error.

**Simple Explanation:** Find direction that reduces loss, take step there.

**Process:**
```
1. Compute loss
2. Compute gradient (which way reduces loss?)
3. Update weights: weight = weight - lr × gradient
4. Repeat
```

**Learning Rate:** Controls step size.

---

### Backpropagation
**Definition:** Algorithm computing gradients through neural network.

**Simple Explanation:** Work backwards from output to find which weights caused error.

**Process:**
```
Forward Pass: Input → compute output
Backward Pass: Output error → trace back → which weights caused it?
Update: Adjust those weights
```

---

### Learning Rate
**Definition:** Controls how much weights change per update.

**Simple Explanation:** The "step size" when walking toward better solutions.

**Effects:**
```
High (0.1):   Large steps → converges fast but might overshoot
Medium (0.01): Balanced 
Low (0.001):  Small steps → slow but stable convergence
```

**In Context:**
- Inner learning rate (0.01): How quickly to adapt
- Outer learning rate (0.001): How quickly to meta-learn

---

### Batch
**Definition:** Group of samples processed together.

**Simple Explanation:** Process multiple examples at once for efficiency.

**In Context:**
- Support batch: 16 examples
- Query batch: 16 examples

**Benefit:** GPU efficiency and better gradient estimates.

---

### Learning Rate Schedule / Scheduler
**Definition:** Plan for how to change learning rate over time.

**Common Schedules:**
- Step decay: Reduce every N epochs
- Exponential decay: Gradual reduction
- Cosine annealing: Smooth curve (our default)

**Why?** Early training needs larger steps, later training needs smaller steps.

---

### Regularization
**Definition:** Techniques to prevent overfitting.

**Common Techniques:**
- Dropout: Randomly disable neurons
- L1/L2: Penalize large weights
- Early stopping: Stop when validation stops improving
- Data augmentation: Vary training data

**In Context:** We use data augmentation and gradient clipping.

---

### Overfitting
**Definition:** Model learns training data too well, fails on new data.

**Simple Explanation:** Memorizing answers instead of learning patterns.

**Symptom:**
```
Training accuracy: 95%
Validation accuracy: 60%
→ Big gap = overfitting
```

**Solution:** Regularization, more data, smaller model.

---

### Underfitting
**Definition:** Model too simple to learn the problem.

**Symptom:**
```
Training accuracy: 50%
Validation accuracy: 50%
→ Both bad = underfitting
```

**Solution:** Larger model, more features, more capacity.

---

## Evaluation Protocols

### Zero-Shot Evaluation
**Definition:** Testing on new domain without any adaptation.

**Simple Explanation:** Pure generalization (does model work as-is?).

**In Context:** Test on ETIS-LaribPolypDB without using 16 examples for adaptation.

**Metric:** How well trained model generalizes.

---

### Few-Shot Evaluation
**Definition:** Testing with quick adaptation on few examples.

**Simple Explanation:** Get 16 examples, adapt 5 steps, then test.

**In Context:** Measure how quickly model adapts.

**Metric:** How fast model adapts.

---

### Cross-Validation
**Definition:** Train/test on different data splits.

**In Context:** We don't use k-fold CV. Instead, we have fixed train/test split (3 train hospitals, 2 test hospitals).

---

## Data Augmentation

### Data Augmentation / Augmentation
**Definition:** Artificially creating training variations.

**Simple Explanation:** Flip/rotate/distort images to make model robust.

**In Context:** Used in `data/augmentations.py`.

**Common Augmentations:**
- Rotation: ±45 degrees
- Flip: Horizontal/vertical
- Color jitter: Adjust brightness/contrast
- Elastic deformation: Slight shape distortion

**Why?** Prevents overfitting, improves generalization.

---

### Augmentation Pipeline
**Definition:** Sequence of augmentations applied to each sample.

**Process:**
```
Input Image → Rotate → Flip → Color Jitter → Elastic Deform → Output
(randomized, different each epoch)
```

---

## Numerics & Stability

### Gradient Clipping
**Definition:** Limit gradient magnitude to prevent explosions.

**Formula:**
```
if ||gradient|| > threshold:
    gradient = gradient × threshold / ||gradient||
```

**Why?** Prevents NaN values, enables stable training.

**In Context:** grad_clip: 1.0 in config.

---

### Learning Rate Warmup
**Definition:** Gradually increase learning rate at start of training.

**Why?** Cold start with high LR causes instability.

**Process:**
```
Epoch 1: LR = 0.0001 (very small)
Epoch 2: LR = 0.0002
...
Epoch 10: LR = 0.001 (target LR reached)
```

---

### Batch Normalization
**Definition:** Normalize features within each batch.

**Effect:** Stabilizes training, allows higher learning rates.

**In Context:** Applied in encoder and decoder.

---

## Miscellaneous

### Hyperparameter
**Definition:** Setting that controls training (not learned).

**Examples:**
- Learning rate (set by us)
- Batch size (set by us)
- Number of layers (set by us)

**Distinguished From:**
- Parameters: Learned from data (weights)
- Hyperparameters: Set before training (learning rate)

---

### Ablation Study
**Definition:** Remove components one-by-one to measure importance.

**Example:**
```
Test 1: Full model → 78% accuracy
Test 2: Remove FAW → 72% accuracy
Test 3: Remove meta-learning → 65% accuracy

Conclusion: Meta-learning adds 13%, FAW adds 6%
```

---

### Reproducibility / Seed
**Definition:** Using fixed random seed to get same results.

**In Context:** `seed: 42` in config ensures reproducible training.

**Why?** Neural networks have random initialization and data shuffling.

---

### Validation Set
**Definition:** Data used to evaluate during training (not for learning).

**Simple Explanation:** Separate test set used to decide when to save best model.

**In Context:** Last hospital in training set split as validation.

---

### Test Set
**Definition:** Data used for final evaluation after training.

**Simple Explanation:** Data model has never seen.

**In Context:** Two unseen hospitals (ETIS-LaribPolypDB, CVC-300).

---

### Checkpoint
**Definition:** Saved model state at specific training point.

**In Context:** Save every 10 epochs, can resume from any checkpoint.

---

### Convergence
**Definition:** Training process reaching stability (loss stops decreasing).

**Simple Explanation:** Model has learned as much as possible.

**In Context:** GMLFNet typically converges after 150-200 epochs.

---

## Summary Table

| Term | Definition | Key Context |
|------|-----------|-------------|
| Polyp | Abnormal tissue growth | Medical: what we detect |
| Segmentation | Pixel-wise classification | What model outputs |
| Domain | Data style/distribution | Different hospitals |
| Meta-Learning | Learning to learn | Core approach |
| MAML | Two-loop meta-algorithm | Training algorithm |
| FAW | Lightweight adaptation module | Key innovation |
| FiLM | Feature modulation | Adaptation mechanism |
| Encoder | Feature extractor | Backbone network |
| Decoder | Feature reconstructor | Output generator |
| Dice | Overlap metric | Primary evaluation |
| Zero-shot | No adaptation | Tests generalization |
| Few-shot | With adaptation | Tests adaptability |

---

**Fully completed! You now understand the terminology! 🎉**
