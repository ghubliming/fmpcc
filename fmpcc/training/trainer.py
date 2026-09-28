import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def cosine_with_warmup(optimizer, warmup, total):
    # as diffusers.optimization.get_cosine_schedule_with_warmup (half a cosine period)
    def factor(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    """Adam with cosine schedule; EMA of the weights; held-out split for the best checkpoint;
    periodic checkpoints every steps/checkpoints steps plus the final one; manifest.json beside them."""

    def __init__(self, model, dataset, cfg, logdir, device='cuda', meta=None, wandb_run=None):
        t, o = cfg['training'], cfg['objective']
        self.model, self.cfg, self.logdir, self.device = model, cfg, logdir, device
        self.meta = meta or {}
        self.wandb = wandb_run
        self.n_steps = int(t['steps'])
        self.accumulate = int(o['accumulate'])
        self.gradient_clip = float(o.get('gradient_clip', 0.0))
        self.ema_decay, self.ema_start, self.ema_every = float(t['ema_decay']), int(t['ema_start']), int(t['ema_every'])
        self.log_every, self.test_batches = int(t['log_every']), int(t['test_batches'])
        self.save_every = self.n_steps // int(t['checkpoints'])
        self.ema_model = copy.deepcopy(model)
        n_train = int(t['train_split'] * len(dataset))
        train_set, test_set = torch.utils.data.random_split(
            dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(int(t['split_seed'])))
        loader = lambda d: cycle(torch.utils.data.DataLoader(
            d, batch_size=int(o['batch_size']), num_workers=int(t['workers']), shuffle=True, pin_memory=True))
        self.train_loader, self.test_loader = loader(train_set), loader(test_set)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(o['lr']))
        self.scheduler = cosine_with_warmup(self.optimizer, int(t['lr_warmup']), self.n_steps)
        self.step = 0
        self.best = np.inf
        self.manifest = {'checkpoints': [], 'log': []}
        self.ema_model.load_state_dict(self.model.state_dict())
        os.makedirs(logdir, exist_ok=True)

    def _ema(self):
        if self.step < self.ema_start:
            self.ema_model.load_state_dict(self.model.state_dict())
            return
        for cur, ma in zip(self.model.parameters(), self.ema_model.parameters()):
            ma.data = ma.data * self.ema_decay + (1 - self.ema_decay) * cur.data

    def _batch(self, loader):
        x, cond = next(loader)
        return to_device(x, self.device), to_device(cond, self.device)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        total = 0.
        for _ in range(self.test_batches):
            loss, _ = self.model.loss(*self._batch(self.test_loader))
            total += (loss / self.accumulate).item()
        self.model.train()
        return total / self.test_batches

    def train(self):
        if self.step >= self.n_steps:
            return
        self.model.train()
        t0 = time.time()
        while self.step < self.n_steps:
            if hasattr(self.model, 'train_step'):
                self.model.train_step = self.step
            for _ in range(self.accumulate):
                loss, info = self.model.loss(*self._batch(self.train_loader))
                loss = loss / self.accumulate
                loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.step % self.ema_every == 0:
                self._ema()
            if self.step % self.save_every == 0:
                self.save(periodic=True)
            if self.step % self.log_every == 0:
                test_loss = self.test()
                entry = {'step': self.step, 'loss': loss.item(), 'held_out_loss': test_loss,
                         'lr': self.scheduler.get_last_lr()[0], 'time_s': round(time.time() - t0, 1)}
                entry.update({k: float(v) for k, v in info.items() if k != 'loss'})
                self.manifest['log'].append(entry)
                if test_loss < self.best:
                    self.best = test_loss
                    self.save(name='best')
                self._write_manifest()
                print('[ train ] ' + '  '.join(f'{k}={v:.5g}' if isinstance(v, float) else f'{k}={v}'
                                                for k, v in entry.items()), flush=True)
                if self.wandb is not None:
                    self.wandb.log(entry, step=self.step)
            self.step += 1
        self.save()

    def _payload(self, full):
        data = {'step': self.step, 'model': self.model.state_dict(), 'ema': self.ema_model.state_dict(),
                'config': self.cfg, **self.meta}
        if full:
            data.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), best=self.best)
        return data

    def save(self, name=None, periodic=False):
        name = name or f'{self.step:06d}'
        path = os.path.join(self.logdir, f'checkpoint_{name}.pt')
        torch.save(self._payload(full=name != 'best'), path + '.tmp')
        os.replace(path + '.tmp', path)
        if periodic:
            for old in glob.glob(os.path.join(self.logdir, 'checkpoint_[0-9]*.pt')):
                if old != path:
                    os.remove(old)
        held_out = self.manifest['log'][-1]['held_out_loss'] if self.manifest['log'] else None
        self.manifest['checkpoints'] = [c for c in self.manifest['checkpoints']
                                        if c['file'] != os.path.basename(path) and os.path.exists(os.path.join(self.logdir, c['file']))]
        self.manifest['checkpoints'].append({'file': os.path.basename(path), 'step': self.step,
                                             'held_out_loss': held_out if name == 'best' else None,
                                             'saved': time.strftime('%Y-%m-%d %H:%M:%S')})
        self._write_manifest()

    def _write_manifest(self):
        with open(os.path.join(self.logdir, 'manifest.json'), 'w') as f:
            json.dump(self.manifest, f, indent=1)

    def resume(self):
        steps = sorted(int(os.path.basename(p)[len('checkpoint_'):-3])
                       for p in glob.glob(os.path.join(self.logdir, 'checkpoint_[0-9]*.pt')))
        if not steps:
            return False
        data = torch.load(os.path.join(self.logdir, f'checkpoint_{steps[-1]:06d}.pt'), map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.best = data['best']
        self.step = data['step'] + 1 if data['step'] < self.n_steps else data['step']
        path = os.path.join(self.logdir, 'manifest.json')
        if os.path.exists(path):
            with open(path) as f:
                self.manifest = json.load(f)
        print(f'[ train ] resumed at step {self.step}', flush=True)
        return True
