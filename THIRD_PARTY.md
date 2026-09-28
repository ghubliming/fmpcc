# Third-party code and models

| where | what | source | licence |
| :-- | :-- | :-- | :-- |
| `fmpcc/envs/d3il/` | D3IL simulation framework (MuJoCo backend), the obstacle-avoidance and alignment environments, the Panda and object models, the alignment split and context files | D3IL, KIT Autonomous Learning Robots Lab, commit `1d9c7185` | MIT, `fmpcc/envs/d3il/LICENSE` |
| `fmpcc/envs/quadrotor/model/` | Skydio X2 model, mesh and texture; `quadrotor_modified.xml` is the Menagerie `x2.xml` with the quadrotor-task patch of MuJoCo MPC (body orientation set, IMU sensors and hover keyframe removed); the scene files are written for this code | MuJoCo Menagerie, MuJoCo MPC | Apache-2.0, `fmpcc/envs/quadrotor/model/LICENSE-skydio_x2.txt` |
| `fmpcc/envs/quadrotor/predictive_sampling.py` | predictive-sampling planner (MJX) | MuJoCo MPC, DeepMind | Apache-2.0, file header |

Changes to the vendored D3IL code: the import root is `fmpcc.envs.d3il`; the PyBullet, mujoco-py and SL backends,
the gym registrations, the point-cloud helper and one test module are removed; temporary model files are written
with flush and sync and deleted at process exit; the logger runs without W&B installed. The predictive-sampling
module drops `mpc_rollout` and its brax import.

Adapted from the code of DPCC (Römer et al., 2025), itself built on Diffuser (Janner et al., 2022): the temporal
U-Net, the diffusion model, the dataset windows, the trainer, the projection program and the per-step projection.
The DPCC code ships without a licence file. The image encoder follows D3IL / robomimic (ResNet-18 with spatial
softmax). Implemented after the papers: flow matching (Lipman et al., 2023), MeanFlow (Geng et al., 2025), α-Flow
(Zhang et al., 2025) and endpoint projection (HardFlow, Li et al., 2025, Algorithm 1 with zero cost).
