import numpy as np

def gini_coefficient(item_exposure: np.ndarray) -> float:
    """
    Measures inequality of item exposure across recommendations.
    0.0 = perfectly fair (all items exposed equally)
    1.0 = completely unfair (one item gets all exposure)
    """
    x = np.sort(np.array(item_exposure, dtype=float))
    x += 1e-9                          # avoid division by zero
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def compute_exposure(recs_dict: dict, n_items: int, k: int = 10) -> np.ndarray:
    """
    Count how many times each item appears in the top-K lists.
    recs_dict: {user_id: [item_id, ...]}
    """
    exposure = np.zeros(n_items, dtype=float)
    for topk in recs_dict.values():
        for item in topk[:k]:
            if item < n_items:
                exposure[item] += 1.0
    return exposure


def coverage(recs_dict: dict, n_items: int, k: int = 10) -> float:
    """
    Catalog coverage: fraction of all items that appear in at least one top-K list.
    Higher is better — means the model recommends diverse items.
    """
    recommended = set()
    for topk in recs_dict.values():
        recommended.update(topk[:k])
    return len(recommended) / n_items


def novelty(recs_dict: dict, item_popularity: np.ndarray, k: int = 10) -> float:
    """
    Average novelty of recommended items.
    Popular items have low novelty; long-tail items have high novelty.
    item_popularity: array of interaction counts per item.
    """
    n_users = len(recs_dict)
    total_novelty = 0.0
    pop_sum = item_popularity.sum()
    for topk in recs_dict.values():
        for item in topk[:k]:
            p = item_popularity[item] / pop_sum if pop_sum > 0 else 1e-9
            total_novelty += -np.log2(p + 1e-9)
    return total_novelty / (n_users * k)


def popularity_bias(recs_dict: dict, item_popularity: np.ndarray, k: int = 10) -> float:
    """
    Average popularity of recommended items.
    Lower is better — means the model avoids always recommending popular items.
    """
    n_users = len(recs_dict)
    total_pop = 0.0
    for topk in recs_dict.values():
        for item in topk[:k]:
            total_pop += item_popularity[item]
    return total_pop / (n_users * k)


def compute_all(recs_dict: dict, n_items: int,
                item_popularity: np.ndarray = None, k: int = 10) -> dict:
    """
    Compute all fairness metrics at once and return as a dict.
    Use this to produce the comparison table in your paper.
    """
    exposure = compute_exposure(recs_dict, n_items, k)
    results = {
        'gini':     round(gini_coefficient(exposure), 4),
        'coverage': round(coverage(recs_dict, n_items, k), 4),
    }
    if item_popularity is not None:
        results['novelty']          = round(novelty(recs_dict, item_popularity, k), 4)
        results['popularity_bias']  = round(popularity_bias(recs_dict, item_popularity, k), 4)
    return results