import argparse
import os
import random

import numpy as np
import torch

from fmpcc import config, models
from fmpcc.data import build_dataset
from fmpcc.training.trainer import Trainer


def run_name(cfg):
    o, model = cfg['objective'], cfg['model']
    if model == 'diffusion':
        return f'diffusion_K{o["steps"]}'
    if model == 'ci_meanfm':
        return f'ci_meanfm_ae{o["alpha_end"]}'
    return model


def main():
    p = argparse.ArgumentParser(description='Train one model for one seed.')
    p.add_argument('config')
    p.add_argument('--model', required=True, choices=models.MODELS)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    p.add_argument('--logdir', default='logs')
    p.add_argument('--tag', default='')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()

    cfg = config.select_model(config.load(args.config, args.set), args.model)
    cfg['seed'] = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = build_dataset(cfg)
    visual = cfg['environment'] == 'aligning'
    model = models.build(args.model, cfg['objective'], dataset.observation_dim, dataset.action_dim,
                         cfg['dataset']['horizon'], goal_dim=dataset.goal_dim, visual=visual,
                         train_steps=int(cfg['training']['steps'])).to('cuda')
    name = run_name(cfg) + (f'_{args.tag}' if args.tag else '')
    logdir = os.path.join(args.logdir, cfg['environment'] if cfg['environment'] != 'uav' else f'uav_{cfg["dataset"]["scene"]}',
                          name, f'seed_{args.seed}')
    config.save(cfg, os.path.join(logdir, 'config.yaml'))
    meta = {'model_name': args.model, 'observation_dim': dataset.observation_dim, 'action_dim': dataset.action_dim,
            'goal_dim': dataset.goal_dim, 'horizon': cfg['dataset']['horizon'], 'visual': visual,
            'normalizer': getattr(dataset, 'raw_normalizer', dataset.normalizer).state()}
    run = None
    if args.wandb or cfg['training'].get('wandb', False):
        import wandb
        run = wandb.init(project=cfg['training'].get('wandb_project', 'fmpcc'), name=f'{name}-seed{args.seed}',
                         config=cfg, dir=logdir)
    print(f'[ train ] {args.model} seed {args.seed}: {len(dataset)} windows, goal_dim {dataset.goal_dim} -> {logdir}', flush=True)
    trainer = Trainer(model, dataset, cfg, logdir, meta=meta, wandb_run=run)
    if args.resume:
        trainer.resume()
    trainer.train()
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
