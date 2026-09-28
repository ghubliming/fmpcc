import os
import random
import time

import numpy as np
import torch

from ..data.normalizer import widen
from ..envs.quadrotor import scenes
from ..envs.quadrotor.controller import GeometricController
from ..envs.quadrotor.paths import reference
from ..planning import Planner, guiding_steps, skip_reason
from ..projection.constraints import uav_set, uav_violations
from ..projection.program import Projector
from . import checkpoint, figure, results
from .avoiding import parse, ragged, variant_name

DATASET_HZ = 33
GOAL_RADIUS = 0.30
ENVELOPE = {'corridor': ((-2.8, -0.45, 0.70), (2.8, 0.45, 1.30)), 's_curve': ((-3.2, -1.25, 0.70), (3.2, 1.25, 1.30))}
ENVELOPE_SLACK, SPEED_MAX, MAP_XY, MAP_Z = 2.0, 6.0, 10.0, 10.0


def diverged(p, v, q, scene):
    """Reason the vehicle is lost (non-finite, off the map, off the demonstrated envelope ± 2 m, faster than
    6 m/s, upside down) or None."""
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
        return 'nan_state'
    if np.any(np.abs(p[:2]) > MAP_XY) or p[2] > MAP_Z or p[2] < -0.5:
        return 'off_map'
    lo = np.array(ENVELOPE[scene][0], dtype=float) - ENVELOPE_SLACK
    hi = np.array(ENVELOPE[scene][1], dtype=float) + ENVELOPE_SLACK
    if np.any(p < lo) or np.any(p > hi):
        return 'off_route'
    if float(np.linalg.norm(v)) > SPEED_MAX:
        return 'overspeed'
    if np.all(np.isfinite(q)) and 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]) < 0.0:
        return 'inverted'
    return None


def flight(model, cfg, planner, build_projector, lane, trial_seed):
    import mujoco
    scene = cfg['scene']
    traj, start, duration = reference(scene, lane, np.random.default_rng(trial_seed))
    goal = np.asarray(traj(duration)[0], dtype=float)
    heading = (goal - np.asarray(traj(max(duration - 0.1, 0.0))[0], dtype=float))[:2]
    heading = heading / np.linalg.norm(heading) if np.linalg.norm(heading) > 1e-9 else np.array([1.0, 0.0])
    data = mujoco.MjData(model)
    data.qpos[:3] = start
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if cfg['controller'] == 'predictive_sampling':
        from ..envs.quadrotor.tracker import PredictiveSamplingTracker
        tracker = PredictiveSamplingTracker(model)
    else:
        tracker = GeometricController(model)
    dt = float(model.opt.timestep)
    decim = max(1, int(round(1.0 / (dt * DATASET_HZ))))
    limit = int(cfg['episode_limit'])
    p_des = np.asarray(start, dtype=float).copy()
    obs_traj, act_traj, plans, executed, times, proj_times = [], [], [], [], [], []
    hits = n_phys = 0
    min_z = float('inf')
    finish = goal_line = reached = False
    abort, steps_run = None, limit
    for k in range(limit):
        p = data.qpos[:3].copy()
        obs = np.concatenate([p_des, p]).astype(np.float32)
        if build_projector is not None:
            planner.set_projector(build_projector(float(p_des[0]) if cfg['bind'] == 'setpoint' else float(p[0])))
        t0 = time.perf_counter()
        action, record = planner(obs)
        times.append(time.perf_counter() - t0)
        proj_times.append(record['projection_time'])
        action = np.asarray(action, dtype=float).reshape(-1)[:3]
        obs_traj.append(obs)
        act_traj.append(action.astype(np.float32))
        plans.append(record['observations'].astype(np.float32))
        executed.append(record['executed'])
        p_des = p_des + action
        v_des = action / (1.0 / DATASET_HZ) if cfg['controller'] == 'predictive_sampling' else np.zeros(3)
        for _ in range(decim):
            data.ctrl[:4] = tracker.compute(data.qpos[:3].copy(), data.qpos[3:7].copy(), data.qvel[:3].copy(),
                                            data.qvel[3:6].copy(), p_des, v_des)
            mujoco.mj_step(model, data)
            n_phys += 1
            hits += int(any(scenes.obstacle_contact(model, data.contact[i]) for i in range(data.ncon)))
            min_z = min(min_z, float(data.qpos[2]))
            dist = float(np.linalg.norm(data.qpos[:3] - goal))
            goal_line = goal_line or float(np.dot(data.qpos[:2] - goal[:2], heading)) >= 0.0 or dist < GOAL_RADIUS
            finish = finish or goal_line or float(data.qpos[0]) >= cfg['finish_x']
            reached = reached or dist < GOAL_RADIUS
        if not reached:
            abort = diverged(data.qpos[:3].copy(), data.qvel[:3].copy(), data.qpos[3:7], scene)
            if abort is not None:
                break
        if reached or k == limit - 1:
            steps_run = k + 1
            break
    safe = hits / max(n_phys, 1) <= scenes.CONTACT_LIMIT[scene] and min_z > 0.2 and abort is None
    n_viol, total = uav_violations(np.asarray(obs_traj, dtype=float)[:, 3:6], cfg['constraints'], cfg['geometry'])
    return {'observations': np.array(obs_traj), 'actions': np.array(act_traj), 'plans': np.array(plans),
            'executed': np.array(executed), 'lane': lane, 'finish': float(finish and safe),
            'finish_and_constraints': float(finish and safe and n_viol == 0),
            'goal_reached': float(reached and safe), 'goal_line': float(goal_line and safe),
            'goal_distance': float(np.linalg.norm(data.qpos[:3] - goal)), 'safe': float(safe),
            'contact_fraction': hits / max(n_phys, 1), 'min_z': min_z, 'violating_steps': float(n_viol),
            'total_violation': float(total), 'violation_free': float(n_viol == 0), 'aborted': abort or '',
            'control_steps': float(steps_run), 'time_per_step': float(np.mean(times)) if times else 0.0,
            'projection_time_per_step': float(np.mean(proj_times)) if proj_times else 0.0}


def draw(cfg):
    def _draw(ax):
        g = cfg['constraints']['geometries'][cfg['geometry']]
        for hs in g['halfspaces']:
            if hs.get('plane', 'xy') == 'xy':
                (x1, y1), (x2, y2) = hs['line']
                ax.plot([x1, x2], [y1, y2], 'b--')
    return _draw


def run(cfg, run_dir, out_dir, device='cuda'):
    import mujoco
    ccfg = cfg['constraints']
    model_mj = mujoco.MjModel.from_xml_path(scenes.xml(cfg['scene_xml']))
    lanes = scenes.LANES[cfg['scene']]
    for seed in cfg['seeds']:
        model, raw, meta = checkpoint.load(os.path.join(run_dir, f'seed_{seed}'), cfg['checkpoint'], cfg['weights'], device)
        normalizer = widen(raw, cfg['constant_margin'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tightened = cfg['tightened']
        for projection, rule in map(parse, cfg['variants']):
            label = variant_name(projection, rule, tightened and projection != 'unguided')
            reason = skip_reason(model, projection, cfg['steps'], cfg['activation_threshold'])
            if reason:
                print(f'[ eval ] skip {label}: {reason}', flush=True)
                continue

            def build(x):
                return Projector(meta['horizon'], 9, 3, uav_set(ccfg, cfg['geometry'], tightened, cfg['bind'], cfg['action_bound'],
                                                                normalizer['actions'], current_x=x), normalizer, ccfg['dt'])
            projector = build(None) if projection != 'unguided' else None
            planner = Planner(model, normalizer, projection, rule, cfg['candidates'], cfg['steps'], cfg['activation_threshold'],
                              projector, goal_dim=None if projection == 'unguided' else 0)
            flights = []
            for i in range(cfg['flights']):
                f = flight(model_mj, cfg, planner, build if projection != 'unguided' else None, lanes[i % len(lanes)], 10000 + i)
                flights.append(f)
                print(f'[ eval ] flight {i} ({f["lane"]}): finish {int(f["finish"])}, violating steps {int(f["violating_steps"])}, '
                      f'{1e3 * f["time_per_step"]:.0f} ms per step' + (f', aborted: {f["aborted"]}' if f['aborted'] else ''), flush=True)
            scalar = ('finish', 'finish_and_constraints', 'goal_reached', 'goal_line', 'goal_distance', 'safe', 'contact_fraction',
                      'min_z', 'violating_steps', 'total_violation', 'violation_free', 'control_steps', 'time_per_step',
                      'projection_time_per_step')
            arrays = {k: np.array([f[k] for f in flights]) for k in scalar}
            arrays.update(lane=np.array([f['lane'] for f in flights]), aborted=np.array([f['aborted'] for f in flights]),
                          observations=ragged([f['observations'] for f in flights]), actions=ragged([f['actions'] for f in flights]),
                          candidate_plans=ragged([f['plans'] for f in flights]), executed_candidate=ragged([f['executed'] for f in flights]),
                          network_evaluations=planner.endpoint.nfe if planner.endpoint else 0,
                          guiding_steps=guiding_steps(cfg['steps'], cfg['activation_threshold']) if projection == 'endpoint' else 0)
            base = os.path.join(out_dir, cfg['geometry'], f'seed_{seed}', label)
            resolved = dict(cfg, seed=seed, projection=projection, rule=rule, checkpoint_path=meta['path'],
                            checkpoint_step=int(meta['step']))
            results.write(base, arrays, resolved)
            print(f'[ eval ] {cfg["geometry"]} seed {seed} {label}: finish {int(arrays["finish"].sum())}/{len(flights)}, '
                  f'violation-free {int(arrays["violation_free"].sum())}/{len(flights)}, '
                  f'{1e3 * np.mean(arrays["time_per_step"]):.0f} ms per step', flush=True)
            figure.foresight(base + '.png', [(f['observations'][:, 3:5], [p[:, :, 3:5] for p in f['plans']]) for f in flights],
                             draw(cfg), cfg['limits'], max(1, meta['horizon'] // 2), title=f'{cfg["geometry"]} {label}')
