import os


def build_dataset(cfg, root='.'):
    d = cfg['dataset']
    path = os.path.join(root, d['root'])
    env = cfg['environment']
    if env == 'avoiding':
        from .avoiding import load_episodes
        from .windows import PlanWindows
        return PlanWindows(load_episodes(path), d['horizon'], d['max_path_length'])
    if env == 'aligning':
        from .aligning import AligningWindows
        return AligningWindows(path, d['horizon'])
    if env == 'uav':
        from .uav import load_episodes
        from .windows import PlanWindows
        return PlanWindows(load_episodes(os.path.join(path, d['scene'])), d['horizon'], d['max_path_length'][d['scene']],
                           constant_margin=d['constant_margin'])
    raise KeyError(f'unknown environment {env!r}')
