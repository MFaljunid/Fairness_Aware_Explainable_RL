import numpy as np


def gini_coefficient(item_exposure: np.ndarray) -> float:
    """
    Measures inequality of item exposure across recommendations.

    0.0 = perfectly fair (all items exposed equally)
    1.0 = completely unfair (one item gets all exposure)
    """
    x = np.array(item_exposure, dtype=float)

    if x.sum() == 0:
        return 0.0

    x     = np.sort(x)
    n     = len(x)
    index = np.arange(1, n + 1)

    gini  = (2.0 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return float(np.clip(gini, 0.0, 1.0))


def compute_exposure(recs_dict: dict, n_items: int, k: int = 10) -> np.ndarray:
    """
    Count how many times each item appears across all top-K lists.
    """
    exposure = np.zeros(n_items, dtype=float)
    for topk in recs_dict.values():
        for item in topk[:k]:
            item = int(item)                   # fix: force int — CSV may give float
            if 0 <= item < n_items:
                exposure[item] += 1.0
    return exposure


def coverage(recs_dict: dict, n_items: int, k: int = 10) -> float:
    """
    Catalog coverage: fraction of all items recommended to at least one user.
    Higher is better.
    """
    if n_items == 0:
        return 0.0
    recommended = set()
    for topk in recs_dict.values():
        for item in topk[:k]:
            item = int(item)                   # fix: force int
            if 0 <= item < n_items:
                recommended.add(item)
    return len(recommended) / n_items


def novelty(recs_dict: dict, item_popularity: np.ndarray,
            k: int = 10, normalize: bool = True) -> float:
    """
    Average self-information of recommended items.

    novelty(i) = -log2( p(i) )   where p(i) = interactions(i) / total

    normalize=True divides by log2(n_items) so the result is in [0,1]
    and comparable across datasets with different catalog sizes.
    Higher is better.
    """
    if len(recs_dict) == 0:
        return 0.0

    pop_sum = item_popularity.sum()
    if pop_sum == 0:
        return 0.0

    total_novelty = 0.0
    total_count   = 0

    for topk in recs_dict.values():
        for item in topk[:k]:
            item = int(item)
            if 0 <= item < len(item_popularity):
                p              = item_popularity[item] / pop_sum
                total_novelty += -np.log2(p + 1e-9)
                total_count   += 1

    if total_count == 0:
        return 0.0

    raw = total_novelty / total_count

    # Normalize by log2(n_items) so result is dataset-independent
    if normalize:
        n_items = len(item_popularity)
        raw     = raw / np.log2(n_items + 1e-9)

    return float(raw)


def popularity_bias(recs_dict: dict, item_popularity: np.ndarray,
                    k: int = 10) -> float:
    """
    Average popularity of recommended items.
    Lower is better — means model recommends long-tail items.
    """
    if len(recs_dict) == 0:
        return 0.0

    total_pop   = 0.0
    total_count = 0

    for topk in recs_dict.values():
        for item in topk[:k]:
            item = int(item)
            if 0 <= item < len(item_popularity):
                total_pop   += float(item_popularity[item])
                total_count += 1

    return total_pop / total_count if total_count > 0 else 0.0


def intra_list_diversity(recs_dict: dict,
                         item_embeddings: np.ndarray,
                         k: int = 10) -> float:
    """
    Average pairwise cosine distance between items in each top-K list.
    Higher = more diverse recommendations.

    Denominator is always len(recs_dict) so the metric is comparable
    across models even if some users have short lists.
    """
    if len(recs_dict) == 0:
        return 0.0

    # Pre-normalise embeddings once
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-9
    embs  = item_embeddings / norms

    total_div = 0.0
    n_users   = len(recs_dict)             # fix: always all users

    for topk in recs_dict.values():
        items = [int(i) for i in topk[:k]
                 if 0 <= int(i) < len(item_embeddings)]
        if len(items) < 2:
            total_div += 0.0               # contributes 0, not skipped
            continue
        vecs  = embs[items]                # (m, emb_dim)
        sim   = vecs @ vecs.T             # (m, m) cosine similarity
        m     = len(items)
        pairs = (m * (m - 1)) / 2
        dist  = 1.0 - sim
        total_div += dist[np.triu_indices(m, k=1)].sum() / pairs

    return float(total_div / n_users)


def compute_all(recs_dict: dict,
                n_items: int,
                item_popularity: np.ndarray = None,
                item_embeddings: np.ndarray = None,
                k: int = 10) -> dict:
    """
    Compute all fairness metrics and return as a dict.

    Parameters
    ----------
    recs_dict       : {user_id_str: [item_id, ...]}
    n_items         : total catalog size
    item_popularity : interaction counts per item (optional)
    item_embeddings : item embedding matrix for ILD (optional)
    k               : cutoff for all metrics

    Returns
    -------
    dict ready to print or save to JSON for your paper table.
    """
    if len(recs_dict) == 0:
        return {}

    exposure = compute_exposure(recs_dict, n_items, k)

    results = {
        'k':        k,                                          # fix: always record k
        'n_users':  len(recs_dict),
        'n_items':  n_items,
        'gini':     round(gini_coefficient(exposure), 4),
        'coverage': round(coverage(recs_dict, n_items, k), 4),
    }

    if item_popularity is not None:
        results['novelty']         = round(
            novelty(recs_dict, item_popularity, k, normalize=True), 4)
        results['popularity_bias'] = round(
            popularity_bias(recs_dict, item_popularity, k), 4)

    if item_embeddings is not None:
        results['intra_list_diversity'] = round(
            intra_list_diversity(recs_dict, item_embeddings, k), 4)

    return results