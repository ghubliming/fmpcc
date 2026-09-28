import json
import os

import numpy as np
import yaml


def write(path, arrays, config):
    """<path>.npz with the per-episode arrays; the resolved configuration is stored inside and beside it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False)
    ragged = {k: np.array(v, dtype=object) if isinstance(v, list) else v for k, v in arrays.items()}
    np.savez(path + '.npz', config=text, **ragged)
    with open(path + '.yaml', 'w') as f:
        f.write(text)


def summary(arrays):
    s = {k: float(np.mean(arrays[k])) for k in ('success', 'success_and_constraints', 'constraint_satisfied',
                                                 'violating_steps', 'total_violation', 'time_per_step') if k in arrays}
    ok = np.asarray(arrays['success']) > 0
    s['control_steps_success'] = float(np.mean(np.asarray(arrays['control_steps'])[ok])) if ok.any() else 0.0
    return s


def print_summary(label, arrays):
    print(f'[ eval ] {label}: ' + json.dumps(summary(arrays)), flush=True)
