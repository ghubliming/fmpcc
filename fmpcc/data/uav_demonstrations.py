import os
import pickle

import numpy as np

from ..envs.quadrotor.controller import GeometricController
from ..envs.quadrotor.paths import reference
from ..envs.quadrotor import scenes

DATASET_HZ = 33
OFFSET_SIGMA = 0.02


def fly(scene, lane, seed):
    """One demonstration: the geometric controller tracks the reference path with velocity and acceleration
    feed-forward at every physics step; None when the flight touches an obstacle on more than the scene's
    share of its steps or drops below 0.5 m."""
    import mujoco
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(scenes.xml(scene))
    data = mujoco.MjData(model)
    traj, start, duration = reference(scene, lane, rng)
    data.qpos[:3] = start
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    controller = GeometricController(model)
    dt = float(model.opt.timestep)
    n_steps = int(round(duration / dt))
    steps, hits = [], 0
    for k in range(n_steps):
        p_des, v_des, a_des, yaw = traj(k * dt)
        p, v, q, om = data.qpos[:3].copy(), data.qvel[:3].copy(), data.qpos[3:7].copy(), data.qvel[3:6].copy()
        data.ctrl[:4] = controller.compute(p, q, v, om, p_des, v_des, a_des, yaw)
        mujoco.mj_step(model, data)
        hits += int(any(scenes.obstacle_contact(model, data.contact[i]) for i in range(data.ncon)))
        steps.append({'p': p, 'v': v, 'p_des': np.asarray(p_des, dtype=float), 'q': q.astype(np.float32)})
    if hits / max(n_steps, 1) > scenes.CONTACT_LIMIT[scene] or min(s['p'][2] for s in steps) < scenes.MIN_ALTITUDE:
        return None
    return {'steps': steps, 'dt_physics': dt, 'duration': duration, 'start': start.tolist(), 'contact_fraction': hits / n_steps}


def episode(flight, rng):
    """Every third physics step: observation [p_des, p, v] (9), action Δp_des (3) of the commanded path,
    which carries one constant Gaussian offset per episode (it cancels in the actions)."""
    stride = max(1, round(1.0 / (flight['dt_physics'] * DATASET_HZ)))
    steps = flight['steps'][::stride]
    obs = np.array([np.concatenate([s['p_des'], s['p'], s['v']]) for s in steps], dtype=np.float32)
    targets = np.array([s['p_des'] for s in steps], dtype=np.float32)
    targets = targets + rng.normal(0.0, OFFSET_SIGMA, (1, 3)).astype(np.float32)
    return {'obs': obs, 'actions': np.diff(targets, axis=0).astype(np.float32), 'targets': targets,
            'q': np.array([s['q'] for s in steps], dtype=np.float32),
            'dt': float(flight['dt_physics'] * stride), 'start': flight['start'], 'duration': flight['duration']}


def collect(scene, n_trials, out_dir, seed=0):
    """Demonstrations of one scene: trial i flies lane i mod |lanes| with seed `seed + i`; accepted episodes are
    written as <out_dir>/<scene>_<lane>_<seed>.pkl."""
    os.makedirs(out_dir, exist_ok=True)
    lanes = scenes.LANES[scene]
    saved = rejected = 0
    for i in range(n_trials):
        lane, trial_seed = lanes[i % len(lanes)], seed + i
        flight = fly(scene, lane, trial_seed)
        if flight is None:
            rejected += 1
            continue
        ep = episode(flight, np.random.default_rng(trial_seed + 99991))
        with open(os.path.join(out_dir, f'{scene}_{lane}_pid_default_{trial_seed:07d}.pkl'), 'wb') as f:
            pickle.dump(ep, f, protocol=4)
        saved += 1
        if saved % 50 == 0 or saved == 1:
            print(f'[ collect ] {scene}: saved {saved}, rejected {rejected}', flush=True)
    print(f'[ collect ] {scene}: {saved} episodes, {rejected} rejected -> {out_dir}', flush=True)
