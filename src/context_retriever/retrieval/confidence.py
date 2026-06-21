import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """
    Computes a retrieval confidence score between 0.0 and 1.0.
    Combines semantic search match strength (cosine similarity) and 
    graph-neighborhood layout coherence. Runs deterministically without external API dependencies.
    """
    def score_retrieval(self, seeds: List[Dict[str, Any]], expanded_nodes: List[Dict[str, Any]]) -> float:
        """
        Calculates the weighted, normalized confidence score.
        
        Args:
            seeds: List of search seed element dictionaries containing 'score'.
            expanded_nodes: List of expanded element dictionaries.
            
        Returns:
            A float value between 0.0 and 1.0.
        """
        if not seeds:
            logger.warning("Seeds list is empty. Confidence score defaulted to 0.0.")
            return 0.0

        try:
            # 1. Semantic Component: Average of top 3 cosine similarity scores
            scores = sorted([float(s.get("score", 0.0)) for s in seeds], reverse=True)
            top_scores = scores[:3]
            semantic_comp = sum(top_scores) / len(top_scores) if top_scores else 0.0

            # 2. Graph Component: Coherence of layout neighborhood
            # Normalized ratio of connected layout structures retrieved relative to initial seeds
            graph_comp = 0.0
            seed_count = len(seeds)
            if seed_count > 0:
                # We filter out seed elements from the expanded_nodes count if present
                actual_expansions = [n for n in expanded_nodes if n.get("expansion_reason") != "seed"]
                ratio = len(actual_expansions) / seed_count
                # Cap the layout density ratio contribution at 1.0
                graph_comp = min(1.0, ratio)

            # 3. Weighted Combination
            # We weight semantic match strength at 70% and graph context density at 30%
            weighted_score = (0.7 * semantic_comp) + (0.3 * graph_comp)
            
            # Clamp the output value to ensure it lies strictly in [0.0, 1.0]
            final_score = max(0.0, min(1.0, weighted_score))
            logger.info(
                f"Confidence Score computed: {final_score:.4f} "
                f"(Semantic Component: {semantic_comp:.4f}, Graph Component: {graph_comp:.4f})"
            )
            return final_score

        except Exception as e:
            logger.error(f"Error computing retrieval confidence score: {str(e)}")
            # Default fallback score in case of unexpected calculation issues
            return 0.0
