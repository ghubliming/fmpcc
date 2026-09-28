import os
import random
import time

import numpy as np
import torch

from ..envs import aligning as env_aligning
from ..planning import Planner, guiding_steps, skip_reason
from ..projection.constraints import aligning_set, aligning_violations
from ..projection.program import Projector
from . import checkpoint, figure, results
from .avoiding import parse, ragged, variant_name


def draw(ccfg):
    def _draw(ax):
        import matplotlib
        lb, ub = ccfg['workspace']['lb'], ccfg['workspace']['ub']
        ax.add_patch(matplotlib.patches.Rectangle((lb[0], lb[1]), ub[0] - lb[0], ub[1] - lb[1], fill=False, ls='--'))
        for (p0, p1, side) in ccfg['halfspaces']:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b--')
        for disk in ccfg['disks']:
            ax.add_patch(matplotlib.patches.Circle(disk['center'][:2], disk['radius'], color='b', alpha=0.2))
    return _draw


def rollout(env, planner, context, blocked, candidates, device):
    planner.previous = None
    state, bp, ih = env.reset(random=False, context=context)
    bp, ih = env_aligning.images(bp, ih)
    des, pos = state[:3].copy(), state[:3].copy()
    path, commands, plans, executed, times, abort = [], [], [], [], [], None
    done = False
    while not done:
        path.append(np.asarray(pos, dtype=float).copy())
        action = np.zeros(3)
        if not blocked and abort is None:
            abort = env_aligning.diverged(des, pos)
            if abort is None:
                visual = tuple(torch.from_numpy(img.astype(np.float32)).to(device)[None, None].repeat(candidates, 1, 1, 1, 1)
                               for img in (bp, ih))
                start = time.time()
                action, record = planner(np.concatenate([des, pos]), extra={'visual': visual})
                times.append(time.time() - start)
                plans.append(record['observations'][:, :, 3:6])
                executed.append(record['executed'])
        commands.append(np.asarray(action, dtype=float))
        command = action + des
        obs, _, done, info = env.step(np.concatenate((command, [0, 1, 0, 0]), axis=0))
        if abort is not None:
            break
        des = command[:3]
        pos, bp, ih = obs
        bp, ih = env_aligning.images(bp, ih)
    box_pos = env.scene.get_obj_pos(env.push_box)
    w, x, y, z = [float(v) for v in env.scene.get_obj_quat(env.push_box)]
    target = context[2]
    final = float(np.sqrt((float(box_pos[0]) - target[0]) ** 2 + (float(box_pos[1]) - target[1]) ** 2))
    return {'path': np.array(path), 'commands': np.array(commands), 'plans': np.array(plans), 'executed': np.array(executed),
            'final_distance': final, 'final_angle': float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)))),
            'initial_distance': float(np.linalg.norm(np.array(context[0][:2]) - np.array(target[:2]))),
            'success': float(info['success']), 'in_position': float(final <= env.pos_min_dist),
            'mean_distance': float(info['mean_distance']), 'mode': float(info['mode']),
            'time_per_step': float(sum(times) / max(1, len(times))), 'aborted': abort or ''}


def run(cfg, run_dir, out_dir, device='cuda'):
    ccfg = cfg['constraints']
    contexts = env_aligning.contexts(cfg['context_split'])
    for seed in cfg['seeds']:
        model, normalizer, meta = checkpoint.load(os.path.join(run_dir, f'seed_{seed}'), cfg['checkpoint'], cfg['weights'], device)
        for tightened in cfg['tightened']:
            geometry = cfg['geometry'] + ('_tightened' if tightened else '')
            for projection, rule in map(parse, cfg['variants']):
                label = variant_name(projection, rule, False)
                reason = skip_reason(model, projection, cfg['steps'], cfg['activation_threshold'])
                if reason:
                    print(f'[ eval ] skip {label}: {reason}', flush=True)
                    continue
                projector = Projector(meta['horizon'], 9, 3, aligning_set(ccfg, tightened, normalizer['actions']),
                                      normalizer, ccfg['dt'])
                B = 1 if projection == 'unguided' else cfg['candidates']
                planner = Planner(model, normalizer, projection, rule, B, cfg['steps'], cfg['activation_threshold'],
                                  projector, consistency='full', unnormalize='selected')
                env = env_aligning.make(cfg['episode_limit'])
                env.start()
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                enlarge = ccfg['tightening'] if tightened else 0.0
                rolls = []
                for c in range(cfg['contexts']):
                    blocked = env_aligning.box_blocked(contexts[c], ccfg['disks'], enlarge)
                    r = rollout(env, planner, contexts[c], blocked, B, device)
                    box, half, disk = aligning_violations(r['path'], ccfg)
                    viol = (box > 1e-6) | (half > 1e-6) | (disk > 1e-6)
                    r.update(held=float(blocked), violating_steps=float(viol.sum()), violation_free=float(not viol.any()),
                             workspace_violations=float((box > 1e-6).sum()), halfspace_violations=float((half > 1e-6).sum()),
                             disk_violations=float((disk > 1e-6).sum()), max_disk_penetration=float(disk.max()),
                             control_steps=float(len(r['path'])))
                    rolls.append(r)
                    print(f'[ eval ] context {c}: final distance {r["final_distance"]:.4f} m, violating steps '
                          f'{int(r["violating_steps"])}' + (' (held)' if blocked else '') +
                          (f' (aborted: {r["aborted"]})' if r['aborted'] else ''), flush=True)
                scalar = ('final_distance', 'final_angle', 'initial_distance', 'success', 'in_position', 'mean_distance',
                          'mode', 'time_per_step', 'held', 'violating_steps', 'violation_free', 'workspace_violations',
                          'halfspace_violations', 'disk_violations', 'max_disk_penetration', 'control_steps')
                arrays = {k: np.array([r[k] for r in rolls]) for k in scalar}
                arrays.update(aborted=np.array([r['aborted'] for r in rolls]), observations=ragged([r['path'] for r in rolls]),
                              commands=ragged([r['commands'] for r in rolls]), candidate_plans=ragged([r['plans'] for r in rolls]),
                              executed_candidate=ragged([r['executed'] for r in rolls]),
                              network_evaluations=planner.endpoint.nfe if planner.endpoint else 0,
                              guiding_steps=guiding_steps(cfg['steps'], cfg['activation_threshold']) if projection == 'endpoint' else 0,
                              solver_failures=projector.failures)
                base = os.path.join(out_dir, geometry, f'seed_{seed}', label)
                resolved = dict(cfg, seed=seed, projection=projection, rule=rule, tightened=tightened,
                                checkpoint_path=meta['path'], checkpoint_step=int(meta['step']))
                results.write(base, arrays, resolved)
                print(f'[ eval ] {geometry} seed {seed} {label}: median final distance '
                      f'{np.median(arrays["final_distance"]):.4f} m, violation-free {int(arrays["violation_free"].sum())}'
                      f'/{len(rolls)}, {1e3 * np.mean(arrays["time_per_step"]):.0f} ms per step', flush=True)
                figure.foresight(base + '.png', [(r['path'][:, :2], [p[:, :, :2] for p in r['plans']]) for r in rolls],
                                 draw(ccfg), ccfg['limits'], max(1, meta['horizon'] // 2), title=f'{geometry} {label}')
