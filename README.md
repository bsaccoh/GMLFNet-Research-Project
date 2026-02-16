# GMLFNet: Gradient-Based Meta-Learning with Fast Adaptation Weights for Robust Multi-Centre Polyp Segmentation

A PyTorch implementation of GMLFNet for domain-generalizable polyp segmentation across multiple medical centers using MAML-based meta-learning with Fast Adaptation Weights (FAW).

## Architecture

```
Input Image -> Encoder (Res2Net-50 / PVTv2-B2)
                  |
            Multi-scale Features [f1, f2, f3, f4]
                  |
            Fast Adaptation Weights (FAW)
                  |  -> FiLM modulation (gamma, beta)
                  |
            Multi-Scale Decoder (RFB + Reverse Attention)
                  |
            Segmentation Map
```

**Key Innovation:** The FAW module generates lightweight modulation parameters (~100K params) that are rapidly adapted via MAML inner-loop optimization, enabling fast domain adaptation to unseen medical centers without modifying the heavy encoder/decoder weights.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python data/download.py --output-dir ./datasets
```

## Training

### Meta-Learning Training (GMLFNet)
```bash
python scripts/train_meta.py --config configs/default.yaml
```

### Baseline Training (No Meta-Learning)
```bash
python scripts/train_baseline.py --config configs/default.yaml
```

### Resume from Checkpoint (for Colab sessions)
```bash
python scripts/train_meta.py --config configs/default.yaml --resume runs/checkpoint_epoch50.pth
```

## Evaluation

```bash
# Zero-shot evaluation (no adaptation)
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode zero_shot

# Few-shot evaluation (with adaptation)
python scripts/evaluate.py --checkpoint runs/best_model.pth --mode few_shot
```

## Ablation Studies

```bash
# Run all ablations
python scripts/ablation.py --config configs/default.yaml --ablation all

# Run specific ablation
python scripts/ablation.py --config configs/default.yaml --ablation inner_steps
```

## Datasets

| Dataset | Images | Source |
|---------|--------|--------|
| Kvasir-SEG | 1000 | Vestre Viken Health Trust |
| CVC-ClinicDB | 612 | Hospital Clinic Barcelona |
| CVC-ColonDB | 380 | CVC Barcelona |
| ETIS-LaribPolypDB | 196 | LARIB Clermont-Ferrand |
| CVC-300 | 60 | CVC Barcelona |

## Project Structure

```
GMLFNet/
├── configs/          # YAML configuration files
├── data/             # Dataset loading and episodic sampling
├── models/           # Network architecture (encoder, FAW, decoder)
├── trainers/         # MAML meta-trainer, baseline trainer, evaluator
├── utils/            # Metrics, visualization, logging
├── scripts/          # Training and evaluation entry points
└── notebooks/        # Data exploration and results analysis
```
