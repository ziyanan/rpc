# Setup

For simulation, install SUMO and make sure TraCI is available (either `SUMO_HOME` is set or pass `--sumo-home`).
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Dataset location

Training/eval scripts read the dataset root from `DATASET_DIR`.

Expected files in `DATASET_DIR`:

- `X_clean_train.npy`, `X_clean_val.npy`, `X_clean_test.npy`
- `Y_train.npy`, `Y_val.npy`, `Y_test.npy`
- `timeseries_meta.json`

Example:

```bash
export DATASET_DIR=pytorch_datasets_cali
```

# Training

End-to-end diffusion + predictor:

```bash
USE_CONDITIONED=true \
python net_training/train_net_model.py
```

Two-stage training (stage 1 diffusion, stage 2 predictor):

```bash
USE_CONDITIONED=true \
python net_training/train_net_model_staged.py
```

Transformer baseline training:

```bash
python net_training/baselines/train_transformer_baseline.py
```

Logic-guided transformer baseline training (STL teacher loss):

```bash
USE_ADV_TRAINING=true 
python net_training/baselines/train_transformer_baseline.py
```

Outputs are written under `training_runs/`.

# Evaluation

Diffusion model eval:

```bash
python net_training/test_net_model.py \
  --model training_runs/<run_id>/best_model.pth \
  --attack-type temporal \
  --epsilon 2.0
```

Transformer baseline eval:

```bash
python net_training/baselines/test_transformer_baseline.py \
  --model training_runs/<run_id>/best_model.pth \
  --attack-type temporal \
  --epsilon 2.0
```

# SUMO closed-loop simulation

Collect detector data:

```bash
python -m nyc_simulation.cli collect \
  --sumocfg nyc_simulation/sumo_configs/midtown.sumocfg \
  --out-dir nyc_simulation/data \
  --detectors auto \
  --dt 60 \
  --seed 42
```

Build sliding windows:

```bash
python -m nyc_simulation.cli build \
  --csv nyc_simulation/data/sumo_detector_timeseries.csv \
  --out-dir nyc_simulation/data \
  --seq-len 6 \
  --horizon 3
```

Train on the collected dataset:

```bash
python -m nyc_simulation.cli train \
  --data-dir nyc_simulation/data \
  --use-conditioned true \
  --use-adv-training true \
  --attack-type uniform \
  --attack-prob 0.5 \
  --epochs 50
```

Run closed-loop simulation (certification runs online and gates control):

```bash
python -m nyc_simulation.cli simulate \
  --sumocfg nyc_simulation/sumo_configs/midtown.sumocfg \
  --model-path training_runs/<logic_run>/best_model.pth \
  --baseline-path training_runs/<baseline_run>/best_model.pth \
  --attack-type temporal \
  --epsilon 2.0 \
  --sigma 0.1 \
  --n0 100 \
  --n 1000 \
  --alpha 0.001 \
  --epsilon-cert 0.08 \
  --mode logic \
  --out-dir nyc_simulation/outputs/run1 \
  --seed 42
```

# Certification

Certification in this repo is based on randomized smoothing over STL satisfaction and is used in:

- `nyc_simulation.cli simulate` via `--sigma`, `--n0`, `--n`, `--alpha`, and `--epsilon-cert`

Core implementation lives under `smoothing/` (`STLRandomizedSmoother` and `compute_certified_radius`).

