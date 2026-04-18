import numpy as np
import pandas as pd
from collections import defaultdict


def compute_dp_eo(recs_dict: dict,
                  user_gender: dict,
                  test_items: dict,
                  k: int = 10) -> dict:
    """
    Compute Demographic Parity (DP) and Equal Opportunity (EO).

    These are the same metrics used in FairIR paper (Shi et al.)
    so you can directly compare your results.

    DP: difference in recommendation quality between male/female users
    EO: difference in hit rate between male/female users

    Parameters
    ----------
    recs_dict   : {user_id_str: [item_id, ...]}
    user_gender : {user_id: 'M' or 'F'}
    test_items  : {user_id: correct_test_item}
    k           : cutoff

    Returns
    -------
    dict with DP and EO values
    """
    male_hits   = []
    female_hits = []

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

    male_hr   = np.mean(male_hits)   if male_hits   else 0.0
    female_hr = np.mean(female_hits) if female_hits else 0.0

    dp = abs(male_hr - female_hr)     # Demographic Parity
    eo = abs(male_hr - female_hr)     # Equal Opportunity (same formula for hit rate)

    return {
        'male_HR':   round(male_hr,   4),
        'female_HR': round(female_hr, 4),
        'DP':        round(dp,        4),
        'EO':        round(eo,        4),
        'n_male':    len(male_hits),
        'n_female':  len(female_hits),
    }


def load_user_gender(users_dat_path: str) -> dict:
    """
    Load gender information from MovieLens users.dat file.

    Format: UserID::Gender::Age::Occupation::Zip-code
    Gender: M = male, F = female
    """
    user_gender = {}
    with open(users_dat_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 2:
                user_id = int(parts[0])
                gender  = parts[1]
                user_gender[user_id] = gender
    return user_gender