import numpy as np
import pandas as pd
from collections import defaultdict


def compute_fairir_dp_eo(recs_dict: dict,
                          user_gender: dict,
                          test_set: dict,
                          n_items: int,
                          k: int = 10) -> dict:
    """
    Exact DP and EO from FairIR paper — Equations 19 and 20.
    """
    male_topk   = np.zeros(n_items, dtype=float)
    female_topk = np.zeros(n_items, dtype=float)
    male_eo     = np.zeros(n_items, dtype=float)
    female_eo   = np.zeros(n_items, dtype=float)

    for uid_str, topk in recs_dict.items():
        uid       = int(uid_str)
        gender    = user_gender.get(uid, 'M')
        user_test = test_set.get(uid, set())
        topk_set  = set(int(i) for i in topk[:k])

        for item in topk_set:
            if 0 <= item < n_items:
                if gender == 'M':
                    male_topk[item] += 1
                    if item in user_test:
                        male_eo[item] += 1
                else:
                    female_topk[item] += 1
                    if item in user_test:
                        female_eo[item] += 1

    dp_scores = []
    for v in range(n_items):
        denom = male_topk[v] + female_topk[v]
        if denom > 0:
            dp_scores.append(abs(male_topk[v] - female_topk[v]) / denom)
    dp = float(np.mean(dp_scores)) if dp_scores else 0.0

    eo_scores = []
    for v in range(n_items):
        denom = male_eo[v] + female_eo[v]
        if denom > 0:
            eo_scores.append(abs(male_eo[v] - female_eo[v]) / denom)
    eo = float(np.mean(eo_scores)) if eo_scores else 0.0

    return {'DP': round(dp, 4), 'EO': round(eo, 4)}


def compute_dp_eo(recs_dict: dict,
                  user_gender: dict,
                  test_items: dict,
                  k: int = 10) -> dict:
    """
    Simple DP/EO based on HR difference between groups.
    Kept for backward compatibility with existing files.
    """
    male_hits, female_hits = [], []

    for uid_str, topk in recs_dict.items():
        uid    = int(uid_str)
        gender = user_gender.get(uid, 'M')
        pos    = test_items.get(uid)
        if pos is None:
            continue
        hit = 1.0 if pos in topk[:k] else 0.0
        if gender == 'M':
            male_hits.append(hit)
        else:
            female_hits.append(hit)

    male_hr   = float(np.mean(male_hits))   if male_hits   else 0.0
    female_hr = float(np.mean(female_hits)) if female_hits else 0.0

    return {
        'DP':        round(abs(male_hr - female_hr), 4),
        'EO':        round(abs(male_hr - female_hr), 4),
        'male_HR':   round(male_hr,   4),
        'female_HR': round(female_hr, 4),
        'n_male':    len(male_hits),
        'n_female':  len(female_hits),
    }


def load_user_gender(users_dat_path: str) -> dict:
    """Load gender from MovieLens users.dat"""
    user_gender = {}
    with open(users_dat_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 2:
                user_gender[int(parts[0])] = parts[1]
    return user_gender