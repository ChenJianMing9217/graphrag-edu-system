from typing import List, Dict, Any
from .types import CandidateNode

class Reranker:
    """
    Reranks candidate nodes based on semantic, structural, and contextual signals.
    """
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def rerank(
        self, 
        candidates: List[CandidateNode], 
        user_query: str, 
        config: Dict[str, float],
        task_label: str = None,
        domain_distribution: Dict[str, float] = None
    ) -> List[CandidateNode]:
        if not candidates:
            return []

        # 1. Semantic Scoring (BGE-M3)
        query_vec = self.text_encoder.encode(user_query)
        semantic_weight = config.get("semantic_weight", 0.6)
        structural_weight = config.get("structural_weight", 0.2)

        for cand in candidates:
            node_vec = self.text_encoder.encode(cand.text)
            # Cosine similarity
            cand.score = float(query_vec @ node_vec) * semantic_weight

            # 2. Structural Boosting (Task-based Label Boost)
            if task_label:
                boost_labels = self._get_boost_labels(task_label)
                if cand.label in boost_labels:
                    cand.score += 0.1 * structural_weight

            # 3. Path-Aware Boosting (Domain Distribution Boost)
            if domain_distribution:
                subdomain = cand.properties.get("subdomain")
                if subdomain:
                    prob = domain_distribution.get(subdomain, 0.0)
                    # Boost score based on the probability of the domain it belongs to
                    cand.score += (prob * 0.2) * structural_weight

        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _get_boost_labels(self, task_label: str) -> List[str]:
        mapping = {
            "T_status_query": ["Assessment", "Observation"],
            "T_professional_suggestion": ["Recommendation", "TrainingDirection"],
            "T_overview": ["Summary", "Meta"]
        }
        return mapping.get(task_label, [])
