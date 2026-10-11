"""Illustrate why original event IDs should be split before duplication.

Matches the array construction/split order inspected in the public upstream
reweighter. This is an overlap audit, not a retraining or performance claim.
"""
import json
import numpy as np
from .data import split_indices


def overlap_audit(n=2000, seed=42):
    original = np.arange(n)
    duplicate_ids = np.concatenate([original, original])
    train, valid = split_indices(2*n, seed)
    train_ids, valid_ids = duplicate_ids[train], duplicate_ids[valid]
    original_train, original_valid = split_indices(n, seed)
    corrected_train = np.concatenate([original_train, original_train])
    corrected_valid = np.concatenate([original_valid, original_valid])
    return {'n_original_events': n, 'seed': seed,
            'duplicate_then_split_shared_unique_events': int(len(np.intersect1d(train_ids, valid_ids))),
            'natural_validation_count': int(len(original_valid)),
            'natural_validation_seen_in_duplicate_training': int(np.isin(original_valid, train_ids).sum()),
            'split_events_then_duplicate_shared_unique_events': int(len(np.intersect1d(corrected_train, corrected_valid)))}


if __name__ == '__main__':
    print(json.dumps(overlap_audit(), indent=2))
