import argparse
import os

import yaml

from fmpcc import config


def main():
    p = argparse.ArgumentParser(description='Evaluate one trained run (all seeds) in one environment.')
    p.add_argument('config')
    p.add_argument('--run', required=True, help='training run directory holding seed_<s>/')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    cfg = config.load(args.config, args.set)
    with open(os.path.join(os.path.dirname(args.config), cfg['constraints'])) as f:
        cfg['constraints'] = yaml.safe_load(f)
    with open(os.path.join(args.run, f'seed_{cfg["seeds"][0]}', 'config.yaml')) as f:
        train_cfg = yaml.safe_load(f)
    model = train_cfg['model']
    cfg['model'] = model
    for key in ('checkpoint', 'weights'):
        if isinstance(cfg[key], dict):
            cfg[key] = cfg[key][model]
    if model == 'diffusion':
        cfg['steps'] = train_cfg['objective']['steps']
    label = f'{cfg["name"]}_K{cfg["steps"]}_eta{cfg["activation_threshold"]}_B{cfg["candidates"]}'
    out_dir = os.path.join(args.run, 'eval', label + (f'_{args.tag}' if args.tag else ''))
    print(f'[ eval ] {model} from {args.run} -> {out_dir}', flush=True)
    if cfg['environment'] == 'avoiding':
        from fmpcc.evaluation.avoiding import run
    elif cfg['environment'] == 'aligning':
        from fmpcc.evaluation.aligning import run
    else:
        from fmpcc.evaluation.uav import run
    run(cfg, args.run, out_dir)


if __name__ == '__main__':
    main()
