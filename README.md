# FM-PCC

The code of the thesis *Flow Matching Predictive Control with Constraints*. A generative model — diffusion, flow
matching (FM), Analytic-MeanFM or CI-MeanFM — plans the next steps, a constraint projection makes the plan
feasible, and the first step is executed before the next plan is made. Environments: D3IL obstacle avoidance and
alignment, and three quadrotor scenes.

## Install

```bash
conda create -n fmpcc python=3.10 -y && conda activate fmpcc
conda install -y pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

On a headless machine set `export MUJOCO_GL=egl`. The MuJoCo MPC cells (`controller=predictive_sampling`) need a
second environment: a copy of the first with `pip install -r requirements-mjx.txt`.

## Data

- D3IL: download the dataset of the D3IL repository (ALRhub/d3il) and extract it into `datasets/d3il/`
  (`avoiding/data/`, `aligning/all_data/`).
- Quadrotor: `python scripts/collect_uav_demonstrations.py --scene corridor`, and the same with `--scene s_curve`.

Data kept elsewhere is used with `--set dataset.root=<path>`.

## Train

```bash
python scripts/train.py configs/train/avoiding.yaml --model analytic_meanfm --seed 6
```

Configurations: `avoiding`, `aligning`, `uav` (`--set dataset.scene=s_curve` for the s-curve). Models:
`diffusion`, `fm`, `analytic_meanfm`, `ci_meanfm`. Diffusion fixes its step count at training
(`--set models.diffusion.steps=<K>`). Runs are written under `logs/`; `--wandb` logs to Weights & Biases.

## Evaluate

```bash
python scripts/evaluate.py configs/eval/avoiding.yaml --run logs/avoiding/analytic_meanfm --set steps=2 "seeds=[6]"
```

Configurations: `avoiding`, `aligning`, `uav_pillars`, `uav_corridor_tilt`, `uav_corridor_hump`, `uav_scurve`.
Their defaults are the protocol of the thesis; any value can be changed with `--set key=value`. Every variant
writes its per-episode arrays (`.npz`), the resolved configuration (`.yaml`) and one figure (`.png`) under
`<run>/eval/`.

## Cluster

```bash
slurm/submit.sh slurm/train.sbatch configs/train/avoiding.yaml analytic_meanfm 6
slurm/submit.sh slurm/evaluate.sbatch configs/eval/avoiding.yaml logs/avoiding/analytic_meanfm --set steps=2 "seeds=[6]"
```

Submit from the repository root. `CONDA_ENV` selects the conda
environment, `SBATCH_PARTITION` the partition.

## Layout

```
fmpcc/     models, projection, planning, environments, data, training, evaluation
configs/   training, evaluation and constraint configurations
scripts/   train, evaluate, collect_uav_demonstrations
slurm/     cluster job scripts
data/      supplementary data of the thesis appendix
```

## Licence

MIT, see `LICENSE`. Third-party code keeps its own licence, listed in `THIRD_PARTY.md`.
