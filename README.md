# ch_unet

A U-Net surrogate model for the Cahn-Hilliard (phase-ordering) PDE. The network learns to roll forward one time step of the phase-separation dynamics.

## Contents

- `phase_ordering.py` – numerical solver (`PhaseOrdering`) for the Cahn-Hilliard equation and dataset generation.
- `models.py` – U-Net architecture (periodic convolutions + attention) and PyTorch Lightning training module (`LitUNet`).
- `losses.py` – physics-informed loss combining MSE with a mass-conservation penalty.
- `utilities.py` – dataset loading, mass-enforcement, autoregressive rollout prediction, and length-scale analysis helpers.
- `train.py` – trains a model.
- `pred.py` – runs autoregressive rollouts with a trained checkpoint and saves results.
- `run.sh` – trains and predicts for both `enf_mass=true` and `enf_mass=false`.

## Usage

Generate training data:

```bash
python phase_ordering.py
```

Train a model:

```bash
python train.py --enf_mass true --epochs 20
```

Run predictions from a checkpoint:

```bash
python pred.py --enf_mass true
```

Or run the full train + predict pipeline for both settings:

```bash
./run.sh
```
