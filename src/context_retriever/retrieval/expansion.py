import logging
from typing import List, Dict, Any, Optional, Set
from src import graph
from src.context_retriever.graph.networkx_store import GraphStore

logger = logging.getLogger(__name__)

class GraphExpander:
    """
    Implements controlled layout relation-aware context expansion.
    Uses GraphStore's NetworkX structure to traverse structural document layouts,
    extracting parent headings, table-caption links, cross-references, and local flow.
    """
    def __init__(self, graph_store: Optional[GraphStore] = None):
        """
        Initializes the GraphExpander with a GraphStore instance.
        
        Args:
            graph_store: Optional GraphStore override. If None, instantiates a default.
        """
        self.store = graph_store or GraphStore()
        logger.info("GraphExpander initialized successfully.")

    def _get_namespaced_id(self, document_id: str, element_id: str) -> str:
        """Helper to generate namespaces matching GraphStore convention."""
        return f"{document_id}::{element_id}"

    def expand_context(self, seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expands the list of semantic seed nodes based on logical layout relationships.
        Applies controlled heading, caption, reference, and reading flow rules.
        
        Args:
            seeds: List of search seed element dictionaries (each containing element_id, 
                   document_id, page_number, type, content, score).
                   
        Returns:
            A deduplicated list of unified element dicts containing both seeds and
            expanded elements, decorated with lineage metadata.
        """
        if not seeds:
            return []

        logger.info(f"Initiating context expansion on {len(seeds)} seed nodes...")

        # Maintain a map of namespaced ID to element dictionary
        expanded_pool: Dict[str, Dict[str, Any]] = {}
        
        # 1. Initialize pool with seed nodes, marked as seeds
        for seed in seeds:
            doc_id = seed.get("document_id")
            elem_id = seed.get("element_id")
            if not doc_id or not elem_id:
                continue
                
            namespaced_id = self._get_namespaced_id(doc_id, elem_id)
            
            # Format seed metadata matching the target output contract
            seed_copy = dict(seed)
            seed_copy["expansion_reason"] = "seed"
            seed_copy["source_seed"] = "self"
            
            expanded_pool[namespaced_id] = seed_copy

        # Access underlying networkx graph representation
        graph = self.store.graph

        # 2. Iterate through each seed node to discover relations
        for seed in seeds:
            doc_id = seed.get("document_id")
            elem_id = seed.get("element_id")
            if not doc_id or not elem_id:
                continue
                
            namespaced_seed_id = self._get_namespaced_id(doc_id, elem_id)
            
            # If seed does not exist in graph registry, skip relational expansion
            if not graph.has_node(namespaced_seed_id):
                logger.warning(
                    f"Seed node '{namespaced_seed_id}' not found in GraphStore. Skipping expansion."
                )
                continue

            elem_type = str(seed.get("type", "")).lower()

            # --- Rule 1: Parent Heading Context ---
            # Search predecessors for parent_of edges pointing to this node
            for pred in graph.predecessors(namespaced_seed_id):
                edge_data = graph.get_edge_data(pred, namespaced_seed_id)
                if edge_data and edge_data.get("relation") == "parent_of":
                    self._add_expanded_node(
                        expanded_pool=expanded_pool,
                        node_id=pred,
                        reason="heading_parent",
                        source_seed=namespaced_seed_id
                    )

            # --- Rule 2: Caption ↔ Table / Image Linking ---
            caption_relations = {"captions", "caption_of", "visualized_by"}
            if elem_type == "caption":
                # Find associated table/image nodes (both outgoing and incoming edges)
                for succ in graph.successors(namespaced_seed_id):
                    edge_data = graph.get_edge_data(namespaced_seed_id, succ)
                    if edge_data and edge_data.get("relation") in caption_relations:
                        self._add_expanded_node(
                            expanded_pool=expanded_pool,
                            node_id=succ,
                            reason="caption_link",
                            source_seed=namespaced_seed_id
                        )
                for pred in graph.predecessors(namespaced_seed_id):
                    edge_data = graph.get_edge_data(pred, namespaced_seed_id)
                    if edge_data and edge_data.get("relation") in caption_relations:
                        self._add_expanded_node(
                            expanded_pool=expanded_pool,
                            node_id=pred,
                            reason="caption_link",
                            source_seed=namespaced_seed_id
                        )
            elif elem_type in ("table", "image"):
                # Find associated caption nodes
                for succ in graph.successors(namespaced_seed_id):
                    edge_data = graph.get_edge_data(namespaced_seed_id, succ)
                    if edge_data and edge_data.get("relation") in caption_relations:
                        self._add_expanded_node(
                            expanded_pool=expanded_pool,
                            node_id=succ,
                            reason="caption_link",
                            source_seed=namespaced_seed_id
                        )
                for pred in graph.predecessors(namespaced_seed_id):
                    edge_data = graph.get_edge_data(pred, namespaced_seed_id)
                    if edge_data and edge_data.get("relation") in caption_relations:
                        self._add_expanded_node(
                            expanded_pool=expanded_pool,
                            node_id=pred,
                            reason="caption_link",
                            source_seed=namespaced_seed_id
                        )

            # --- Rule 3: Reference Expansion ---
            ref_relations = {"references_table", "references", "referenced_by"}
            for succ in graph.successors(namespaced_seed_id):
                edge_data = graph.get_edge_data(namespaced_seed_id, succ)
                if edge_data and edge_data.get("relation") in ref_relations:
                    self._add_expanded_node(
                        expanded_pool=expanded_pool,
                        node_id=succ,
                        reason="reference",
                        source_seed=namespaced_seed_id
                    )
            for pred in graph.predecessors(namespaced_seed_id):
                edge_data = graph.get_edge_data(pred, namespaced_seed_id)
                if edge_data and edge_data.get("relation") in ref_relations:
                    self._add_expanded_node(
                        expanded_pool=expanded_pool,
                        node_id=pred,
                        reason="reference",
                        source_seed=namespaced_seed_id
                    )

            # --- Rule 4: Local Reading Flow (Immediate Non-Recursive Expansion) ---
            # Outgoing precedence links
            for succ in graph.successors(namespaced_seed_id):
                edge_data = graph.get_edge_data(namespaced_seed_id, succ)
                if edge_data:
                    relation = edge_data.get("relation")
                    if relation == "precedes":
                        self._add_expanded_node(expanded_pool, succ, "reading_flow", namespaced_seed_id)
                    elif relation == "follows":
                        self._add_expanded_node(expanded_pool, succ, "reading_flow", namespaced_seed_id)
            # Incoming precedence links
            for pred in graph.predecessors(namespaced_seed_id):
                edge_data = graph.get_edge_data(pred, namespaced_seed_id)
                if edge_data:
                    relation = edge_data.get("relation")
                    if relation == "precedes":
                        self._add_expanded_node(expanded_pool, pred, "reading_flow", namespaced_seed_id)
                    elif relation == "follows":
                        self._add_expanded_node(expanded_pool, pred, "reading_flow", namespaced_seed_id)

        # 3. Post-pass: For any table, image, or caption node now in the expanded pool,
        # ensure its corresponding captions/tables/images are also retrieved.
        # This resolves the p_01 -> t_01 -> cap_01 integration path.
        current_keys = list(expanded_pool.keys())
        for namespaced_id in current_keys:
            elem = expanded_pool[namespaced_id]
            elem_type = str(elem.get("type", "")).lower()
            
            if elem_type in ("table", "image", "caption"):
                caption_relations = {"captions", "caption_of", "visualized_by"}
                if elem_type == "caption":
                    for succ in graph.successors(namespaced_id):
                        edge_data = graph.get_edge_data(namespaced_id, succ)
                        if edge_data and edge_data.get("relation") in caption_relations:
                            self._add_expanded_node(
                                expanded_pool=expanded_pool,
                                node_id=succ,
                                reason="caption_link",
                                source_seed=namespaced_id
                            )
                    for pred in graph.predecessors(namespaced_id):
                        edge_data = graph.get_edge_data(pred, namespaced_id)
                        if edge_data and edge_data.get("relation") in caption_relations:
                            self._add_expanded_node(
                                expanded_pool=expanded_pool,
                                node_id=pred,
                                reason="caption_link",
                                source_seed=namespaced_id
                            )
                else:  # table or image
                    for succ in graph.successors(namespaced_id):
                        edge_data = graph.get_edge_data(namespaced_id, succ)
                        if edge_data and edge_data.get("relation") in caption_relations:
                            self._add_expanded_node(
                                expanded_pool=expanded_pool,
                                node_id=succ,
                                reason="caption_link",
                                source_seed=namespaced_id
                            )
                    for pred in graph.predecessors(namespaced_id):
                        edge_data = graph.get_edge_data(pred, namespaced_id)
                        if edge_data and edge_data.get("relation") in caption_relations:
                            self._add_expanded_node(
                                expanded_pool=expanded_pool,
                                node_id=pred,
                                reason="caption_link",
                                source_seed=namespaced_id
                            )

        # 4. Compile output list
        output_list = list(expanded_pool.values())
        logger.info(f"Context expansion complete. Output pool size: {len(output_list)} elements.")
        return output_list

    def _add_expanded_node(
        self, 
        expanded_pool: Dict[str, Dict[str, Any]], 
        node_id: str, 
        reason: str, 
        source_seed: str
    ) -> None:
        """
        Safely extracts a node's full attributes from GraphStore and adds it to 
        the expansion pool if it hasn't already been added.
        
        Args:
            expanded_pool: Map of namespaced IDs to formatted element dictionaries.
            node_id: Namespaced target node ID to fetch.
            reason: Context expansion category logic.
            source_seed: Namespaced node ID of the seed that triggered this traversal.
        """
        # If node is already in the pool, keep the original (seeds or higher-priority reason take precedence)
        if node_id in expanded_pool:
            return

        try:
            metadata = self.store.get_node_metadata(node_id)
            
            # Ensure safe fallback defaults for metadata fields
            page_num = metadata.get("page_number")
            page_num_parsed = int(page_num) if page_num is not None else 1
            content_parsed = str(metadata.get("content") or "")

            # Construct standard schema matching retrieve_seeds contract
            expanded_item = {
                "element_id": metadata.get("raw_id"),
                "document_id": metadata.get("document_id"),
                "page_number": page_num_parsed,
                "type": metadata.get("type"),
                "content": content_parsed,
                "score": 0.0,  # Safe default score for schema consistency
                "expansion_reason": reason,
                "source_seed": source_seed
            }
            
            # Retain extra parsed fields
            for k, v in metadata.items():
                if k not in ["raw_id", "document_id", "page_number", "type", "content"]:
                    expanded_item[k] = v
                    
            expanded_pool[node_id] = expanded_item
            logger.debug(f"Expanded node '{node_id}' via relation '{reason}' from source '{source_seed}'.")
            
        except KeyError:
            logger.warning(f"Attempted to expand node '{node_id}' but it was missing from graph metadata.")
        except Exception as e:
            logger.error(f"Failed to add expanded node '{node_id}': {str(e)}")
