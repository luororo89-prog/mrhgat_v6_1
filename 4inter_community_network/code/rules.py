"""Medical visit screening; grid coordinates are location proxies, never tower fixes."""
import unicodedata
import re
import numpy as np
import pandas as pd

PURPOSES = ['dining', 'lifeservice', 'medical', 'recreation', 'schooling', 'shoping']

def normalize_name(value):
    if pd.isna(value):
        return ''
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value)))

def time_flags(arrival, departure):
    a = pd.to_datetime(arrival, errors='coerce')
    d = pd.to_datetime(departure, errors='coerce')
    minutes = (d - a).dt.total_seconds() / 60
    clock = a.dt.hour * 60 + a.dt.minute + a.dt.second / 60
    # Arrival within a half-open peak window is a project operational definition.
    peak = ((clock >= 480) & (clock < 660)) | ((clock >= 780) & (clock < 960))
    return pd.DataFrame({'minutes': minutes, 'invalid_time': minutes.isna() | (minutes <= 0),
                         'overnight': a.dt.normalize().ne(d.dt.normalize()),
                         'long_stay': minutes >= 480, 'peak_arrival': peak,
                         'weekday': a.dt.dayofweek < 5, 'day': a.dt.day})

def eligible(frame, minimum=30, exclude_work=True):
    mask = (~frame.invalid_time & ~frame.overnight & ~frame.long_stay
            & frame.peak_arrival & frame.minutes.ge(minimum) & ~frame.frequent_presence)
    if exclude_work:
        mask &= ~frame.work_hospital_proxy
    return mask

def distinct_weekdays(presence):
    """One person/facility/day counts once, independent of repeated visits."""
    return (presence.drop_duplicates(['nid', 'facility_id', 'day'])
            .groupby(['nid', 'facility_id']).size().rename('working_days').reset_index())

def split_neighborhood_flow(grouped, source_lookup, target_lookup, ids, attraction):
    """Assign whole records, preserving internal preference and total flow."""
    index = {int(v): i for i, v in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    missing_source = missing_target = 0
    zero_attraction = 0
    for row in grouped.itertuples(index=False):
        source_grid, target_grid, count = row
        if not np.isfinite(count) or count < 0 or count != int(count):
            raise ValueError('Each grid OD count must be a nonnegative integer')
        count = int(count)
        source = source_lookup.get(int(source_grid))
        if source is None:
            missing_source += count
            continue
        targets = target_lookup.get(int(target_grid), [])
        if not targets:
            missing_target += count
            continue
        targets = [source] if source in targets else targets
        if len(targets) == 1:
            weights = np.ones(1)
        else:
            weights = np.asarray([attraction[t] for t in targets],dtype=float)
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Attraction scores must be finite and nonnegative')
            if weights.sum() == 0:
                zero_attraction += count
                continue
            weights = weights / weights.sum()
        shares = count * weights
        allocated = np.floor(shares).astype(np.int64)
        remaining = count - int(allocated.sum())
        if remaining < 0 or remaining > len(targets):
            raise AssertionError('Integer allocation remainder is invalid')
        order = sorted(range(len(targets)), key=lambda i: (-(shares[i] - allocated[i]), int(targets[i])))
        allocated[order[:remaining]] += 1
        for target, assigned in zip(targets, allocated):
            matrix[index[source], index[target]] += assigned
    assert matrix.sum() + missing_source + missing_target + zero_attraction == grouped.iloc[:, 2].sum()
    return matrix, {'missing_source_mass': missing_source, 'missing_target_mass': missing_target,
                    'zero_attraction_mass': zero_attraction,
                    'assigned_mass': int(matrix.sum())}
