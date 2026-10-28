"""Resolve frozen training conditions without sharing fitted representations."""
from pathlib import Path
from evaluation import ROOT


def condition_directory(protocol, condition, override=None, root=ROOT):
    definition = protocol['conditions'][condition]
    if override is not None:
        return Path(override)
    directory = definition.get('data_directory')
    if directory is None:
        if condition != 'wildjailbreak-100k':
            raise ValueError(f'Missing data directory for {condition}')
        directory = 'data/processed/large'
    return Path(root) / directory


def expected_training_rows(protocol, condition):
    return protocol['conditions'][condition]['partitions']['train']['rows']
