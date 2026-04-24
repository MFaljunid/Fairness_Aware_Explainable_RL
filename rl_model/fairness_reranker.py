import numpy as np

class FairnessReranker:
    """
    Post-hoc fairness re-ranking.
    Takes RL scores and re-ranks to balance gender exposure.
    
    Based on MMR (Maximal Marginal Relevance) with fairness constraint.
    """
    
    def __init__(self, n_items: int, lambda_fair: float = 0.5):
        self.n_items     = n_items
        self.lambda_fair = lambda_fair
        
        # Global exposure tracking across all users
        self.male_exposure   = np.zeros(n_items, dtype=np.float32)
        self.female_exposure = np.zeros(n_items, dtype=np.float32)
        self.total_male      = 0
        self.total_female    = 0
    
    def rerank(self, scores: np.ndarray, 
               user_gender: str,
               seen_items: set,
               k: int = 40) -> list:
        """
        Re-rank items balancing relevance and fairness.
        
        Parameters
        ----------
        scores      : (n_items,) relevance scores from RL model
        user_gender : 'M' or 'F'
        seen_items  : items to exclude
        k           : number of items to return
        """
        # Get candidate items sorted by relevance
        candidates = [(item, float(scores[item]))
                      for item in range(self.n_items)
                      if item not in seen_items]
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        # Take top 200 candidates for re-ranking
        candidates = candidates[:200]
        
        selected = []
        for _ in range(k):
            if not candidates:
                break
            
            best_item  = None
            best_score = -float('inf')
            
            for item, rel_score in candidates:
                # Fairness score for this item
                male_c   = float(self.male_exposure[item])
                female_c = float(self.female_exposure[item])
                denom    = male_c + female_c
                
                if denom == 0:
                    fairness_score = 1.0  # never recommended → very fair
                else:
                    dp_item        = abs(male_c - female_c) / denom
                    fairness_score = 1.0 - dp_item  # higher = fairer
                
                # Combined score
                combined = ((1 - self.lambda_fair) * rel_score +
                            self.lambda_fair * fairness_score)
                
                if combined > best_score:
                    best_score = combined
                    best_item  = item
            
            if best_item is None:
                break
                
            selected.append(best_item)
            candidates = [(i, s) for i, s in candidates if i != best_item]
            
            # Update exposure
            if user_gender == 'M':
                self.male_exposure[best_item] += 1
                self.total_male += 1
            else:
                self.female_exposure[best_item] += 1
                self.total_female += 1
        
        return selected
    
    def update(self, recommendations: list, user_gender: str):
        """Update exposure counts after recommendations."""
        for item in recommendations:
            if user_gender == 'M':
                self.male_exposure[item]   += 1
                self.total_male            += 1
            else:
                self.female_exposure[item] += 1
                self.total_female          += 1