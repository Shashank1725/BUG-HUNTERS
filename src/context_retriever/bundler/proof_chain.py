import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ProofChainBuilder:
    """
    Constructs explainability metadata and trace lineage mapping retrieval paths
    back to their semantic entry points.
    """
    def generate_lineage(
        self, 
        seeds: List[Dict[str, Any]], 
        expanded_nodes: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Creates audit trails for all context elements.
        
        Args:
            seeds: List of search seed node dicts.
            expanded_nodes: List of layout expanded node dicts.
            
        Returns:
            A dictionary mapping '{document_id}::{element_id}' to a lineage string description.
        """
        lineage: Dict[str, str] = {}
        
        try:
            # 1. Process seed nodes (Direct semantic matches)
            for seed in seeds:
                doc_id = seed.get("document_id")
                elem_id = seed.get("element_id")
                score = seed.get("score", 0.0)
                if doc_id and elem_id:
                    namespaced_id = f"{doc_id}::{elem_id}"
                    lineage[namespaced_id] = f"[Direct Semantic Match | Score={score:.4f}]"

            # 2. Process expanded nodes (Logical graph derivations)
            for node in expanded_nodes:
                doc_id = node.get("document_id")
                elem_id = node.get("element_id")
                reason = node.get("expansion_reason")
                source = node.get("source_seed")
                
                if doc_id and elem_id:
                    namespaced_id = f"{doc_id}::{elem_id}"
                    
                    # If this node is already registered as a direct seed match, preserve that state
                    if namespaced_id in lineage and lineage[namespaced_id].startswith("[Direct"):
                        continue
                        
                    # Handle nodes returned in expansion lists that are structurally seed nodes
                    if reason == "seed" or source == "self":
                        score = node.get("score", 0.0)
                        lineage[namespaced_id] = f"[Direct Semantic Match | Score={score:.4f}]"
                    else:
                        # Map lineage path: derived via layout connection from trigger seed
                        lineage[namespaced_id] = f"[Derived from {source} via {reason}]"

            logger.info(f"Successfully generated proof lineage for {len(lineage)} retrieved elements.")
            return lineage

        except Exception as e:
            logger.error(f"Failed to generate retrieval lineage audit proof chains: {str(e)}")
            # Return empty lineage dict rather than throwing to avoid breaking context packaging
            return {}
