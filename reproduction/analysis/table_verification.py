"""Strict keyed comparison shared by the two rebuilt table verifiers."""
import numpy as np


def verify_table(actual, reference, keys, counts, values, expected_rows):
    columns=keys+counts+values
    for name, table in [('rebuilt',actual),('reference',reference)]:
        if len(table)!=expected_rows:
            raise ValueError(f'{name}: expected {expected_rows} rows, got {len(table)}')
        if table[columns].isna().any().any() or table.duplicated(keys).any():
            raise ValueError(f'{name}: missing values or duplicate keys')
        if not np.isfinite(table[counts+values].to_numpy()).all():
            raise ValueError(f'{name}: nonfinite values')
    actual=actual.set_index(keys).sort_index()
    reference=reference.set_index(keys).sort_index()
    if not actual.index.equals(reference.index):
        raise ValueError('table keys differ')
    if not np.array_equal(actual[counts].to_numpy(),reference[counts].to_numpy()):
        raise ValueError('pair/road/model counts differ')
    if not np.allclose(actual[values],reference[values],rtol=0,atol=1e-10):
        raise ValueError('estimates or interval endpoints differ (tolerance 1e-10)')
