"""Candidate scores for distinct history interests; history rows are oldest first."""
import numpy as np

METHODS = ('last', 'last2_mean', 'history_max', 'history_top2_mean',
           'max_last_025', 'max_last_050', 'max_last_075')


def interest_scores(similarities, method):
    if similarities.ndim != 2 or len(similarities) < 1:
        raise ValueError('Expected nonempty history x catalog similarities')
    last = similarities[-1]
    if method == 'last':
        return last.copy()
    if method == 'last2_mean':
        return similarities[-2:].mean(axis=0)
    maximum = similarities.max(axis=0)
    if method == 'history_max':
        return maximum
    if method == 'history_top2_mean':
        return np.sort(similarities, axis=0)[-2:].mean(axis=0)
    if method in ('max_last_025', 'max_last_050', 'max_last_075'):
        last_weight = {'max_last_025': .25, 'max_last_050': .5, 'max_last_075': .75}[method]
        return last_weight*last + (1-last_weight)*maximum
    raise ValueError(method)
