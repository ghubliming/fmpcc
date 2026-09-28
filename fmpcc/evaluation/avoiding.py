import os
import time

import numpy as np
import torch

from ..envs.avoiding import make
from ..planning import Planner, guiding_steps, skip_reason
from ..projection.constraints import action_bounds, planar_set, planar_violation
from ..projection.program import Projector
from . import checkpoint, figure, results


def parse(variant):
    """'unguided' | '<per_step|endpoint>:<random|cumulative_cost|temporal_consistency>' -> (projection, rule)."""
    projection, _, rule = variant.partition(':')
    return projection, rule or 'random'


def variants(cfg):
    out = []
    for v in cfg['variants']:
        projection, rule = parse(v)
        out += [(projection, rule, False)] if projection == 'unguided' else [(projection, rule, t) for t in cfg['tightened']]
    return out


def variant_name(projection, rule, tightened):
    return 'unguided' if projection == 'unguided' else f'{projection}_{rule}' + ('_tightened' if tightened else '')


def ragged(items):
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def draw(ccfg, geometry):
    def _draw(ax):
        import matplotlib
        for (p0, p1, side) in ccfg['geometries'][geometry]['halfspaces']:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b--')
        for disk in ccfg['geometries'][geometry]['disks']:
            ax.add_patch(matplotlib.patches.Circle(disk['center'], disk['radius'], color='b', alpha=0.2))
        for c in ccfg['posts']:
            ax.add_patch(matplotlib.patches.Circle(c, ccfg['post_radius'], color='r'))
        ax.axhline(ccfg['goal_y'], color=(0.4, 1, 0.4), linewidth=3)
    return _draw


def episode(env, planner, ccfg, geometry, limit, ad, lower, upper, out, i):
    torch.manual_seed(i)
    obs = env.reset()
    action = env.robot_state()[:2]
    fixed_z = env.robot_state()[2:]
    obs = np.concatenate((action[:2], obs))
    path, acts, plans, executed = [], [], [], []
    for step in range(limit):
        amounts = planar_violation(ccfg, geometry, obs, ad)
        for a in amounts:
            out['total_violation'][i] += a
        if amounts:
            out['constraint_satisfied'][i] = 0
        if step > 0:
            act_obs = np.concatenate((action, obs))
            out['total_violation'][i] += np.sum(np.maximum(0, act_obs - upper)) + np.sum(np.maximum(0, lower - act_obs))
        out['violating_steps'][i] += int(bool(amounts))
        start = time.time()
        action, record = planner(obs)
        out['time_per_step'][i] += time.time() - start
        next_pos_des = action + obs[:2]
        obs, _, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
        success = info[1]
        obs = np.concatenate((next_pos_des[:2], obs))
        # an episode-recording hook (GIF / video frames) would be called here
        plans.append(record['observations'])
        executed.append(record['executed'])
        path.append(obs)
        acts.append(action)
        if success:
            out['success'][i] = 1
        if (terminated or step == limit - 1) and not success:
            out['constraint_satisfied'][i] = 0
        if success or terminated or step == limit - 1:
            out['control_steps'][i] = step
            out['time_per_step'][i] /= step
            if success and out['constraint_satisfied'][i]:
                out['success_and_constraints'][i] = 1
            break
    return np.array(path), np.array(acts), np.array(plans), np.array(executed)


def run(cfg, run_dir, out_dir, device='cuda'):
    ccfg = cfg['constraints']
    idx = ccfg['indices']
    dim = len(idx)
    lower, upper = action_bounds(ccfg['action_bounds'], dim, idx)
    for geometry in cfg['geometries']:
        for seed in cfg['seeds']:
            seed_dir = os.path.join(run_dir, f'seed_{seed}')
            model, normalizer, meta = checkpoint.load(seed_dir, cfg['checkpoint'], cfg['weights'], device)
            ad = meta['action_dim']
            env = make(cfg['plant'])
            env.start()
            for projection, rule, tightened in variants(cfg):
                label = variant_name(projection, rule, tightened)
                reason = skip_reason(model, projection, cfg['steps'], cfg['activation_threshold'])
                if reason:
                    print(f'[ eval ] skip {label}: {reason}', flush=True)
                    continue
                projector = Projector(meta['horizon'], dim, ad, planar_set(ccfg, geometry, tightened), normalizer, ccfg['dt'])
                planner = Planner(model, normalizer, projection, rule, cfg['candidates'], cfg['steps'],
                                  cfg['activation_threshold'], projector)
                n = cfg['episodes']
                out = {k: np.zeros(n) for k in ('success', 'success_and_constraints', 'control_steps', 'violating_steps',
                                                'total_violation', 'time_per_step')}
                out['constraint_satisfied'] = np.ones(n)
                paths, acts, plans, executed = [], [], [], []
                for i in range(n):
                    p, a, c, e = episode(env, planner, ccfg, geometry, cfg['episode_limit'], ad, lower, upper, out, i)
                    paths.append(p), acts.append(a), plans.append(c), executed.append(e)
                arrays = dict(out, observations=ragged(paths), actions=ragged(acts), candidate_plans=ragged(plans),
                              executed_candidate=ragged(executed), network_evaluations=planner.endpoint.nfe if planner.endpoint else 0,
                              guiding_steps=guiding_steps(cfg['steps'], cfg['activation_threshold']) if projection == 'endpoint' else 0,
                              solver_failures=projector.failures)
                base = os.path.join(out_dir, geometry, f'seed_{seed}', label)
                resolved = dict(cfg, geometry=geometry, seed=seed, projection=projection, rule=rule, tightened=tightened,
                                checkpoint_path=meta['path'], checkpoint_step=int(meta['step']))
                results.write(base, arrays, resolved)
                results.print_summary(f'{geometry} seed {seed} {label}', arrays)
                xy = [idx['x'] - ad, idx['y'] - ad]
                figure.foresight(base + '.png', [(p[:, xy], [c[:, :, xy] for c in cs]) for p, cs in zip(paths, plans)],
                                 draw(ccfg, geometry), ccfg['limits'], max(1, meta['horizon'] // 2), title=f'{geometry} {label}')
            env.close()
