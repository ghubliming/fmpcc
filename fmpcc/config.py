import copy
import os

import yaml


def load(path, overrides=()):
    cfg = _read(path)
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'override {item!r} is not key=value')
        _assign(cfg, key.strip(), _value(value))
    return cfg


def _value(text):
    v = yaml.safe_load(text)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def _read(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('base', None)
    if base is not None:
        cfg = merge(_read(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _assign(cfg, dotted, value):
    node = cfg
    keys = dotted.split('.')
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def select_model(cfg, model):
    """Train configuration of one model: the shared keys plus that model's section."""
    if model not in cfg['models']:
        raise KeyError(f'model {model!r} not in {sorted(cfg["models"])}')
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k != 'models'}
    out['model'] = model
    out['objective'] = copy.deepcopy(cfg['models'][model])
    return out


def save(cfg, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
