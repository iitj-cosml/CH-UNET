# CH-UNET

This repository contains code accompanying the paper [Physics-guided Convolutional Neural Network for Domain Growth Prediction in Systems with Conserved Kinetics](https://doi.org/10.48550/arXiv.2606.26128).

## Contents

- `phase_ordering.py` – numerical solver (`PhaseOrdering`) for the Cahn-Hilliard equation and dataset generation.
- `models.py` – U-Net architecture (periodic convolutions + attention) and PyTorch Lightning training module (`LitUNet`).
- `losses.py` – conseravtion loss combining MSE with an order-parameter conservation penalty.
- `utilities.py` – dataset loading, conservation enforcement, autoregressive rollout prediction, and length-scale analysis helpers.
- `train.py` – trains a model.
- `pred.py` – runs autoregressive rollouts with a trained checkpoint and saves results.
- `run.sh` – trains and predicts for both `enf_cons=true` and `enf_cons=false`.

## Usage

Generate training data:

```bash
python phase_ordering.py
```

Train a model:

```bash
python train.py --enf_cons true --epochs 20
```

Run predictions from a checkpoint:

```bash
python pred.py --enf_cons true
```

Or run the full train + predict pipeline for both settings:

```bash
./run.sh
```
