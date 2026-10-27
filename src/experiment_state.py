"""Atomic checkpoints, reproducible random state, and validation early stopping."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_torch(path, state):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def update_validation(state, score, patience, min_delta=0.):
    improved = state['best_score'] is None or score > state['best_score'] + min_delta
    if improved:
        state['best_score'] = score
        state['chosen_epoch'] = state['epoch'] + 1
        state['bad_epochs'] = 0
    else:
        state['bad_epochs'] += 1
    state['early_stopped'] = state['bad_epochs'] >= patience
    return improved
