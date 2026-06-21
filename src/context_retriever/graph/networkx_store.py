import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set
import networkx as nx

logger = logging.getLogger(__name__)

class GraphStore:
    """
    Manages a multi-document layout relationship graph using networkx.DiGraph.
    Prevents ID collision across documents by namespacing element IDs,
    maintains index tables for fast document-scoped lookups, and supports
    custom relational edge queries for context expansion.
    """
    def __init__(self):
        """
        Initializes an empty master directed graph and auxiliary lookups.
        """
        self.graph = nx.DiGraph()
        # Index to lookup all node IDs belonging to a document in O(1)
        self._doc_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        logger.info("Master GraphStore successfully initialized.")

    def _get_namespaced_id(self, document_id: str, element_id: str) -> str:
        """
        Generates a globally unique node ID to prevent collisions between elements
        of different documents.
        
        Args:
            document_id: The document identifier.
            element_id: The element identifier inside the document.
            
        Returns:
            A string in the format '{document_id}::{element_id}'.
        """
        return f"{document_id}::{element_id}"

    def load_document_graph(self, serialized_graph_json: Dict[str, Any], document_id: Optional[str] = None) -> None:
        """
        Loads a document's relationship graph (in NetworkX node-link JSON format)
        and merges it into the master directed graph.
        
        Args:
            serialized_graph_json: Dict representation of the node-link JSON.
            document_id: Optional fallback document ID if not present in the graph root.
            
        Raises:
            ValueError: If document_id is missing or the JSON is invalid.
            RuntimeError: If loading fails.
        """
        # Determine document ID from graph root metadata or parameters
        doc_id = serialized_graph_json.get("graph", {}).get("document_id") or document_id
        if not doc_id:
            error_msg = "Failed to load graph: document_id must be provided in graph metadata or as an argument."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Loading relationship graph for document '{doc_id}'...")

        try:
            # 1. Parse and insert nodes
            nodes_list = serialized_graph_json.get("nodes", [])
            for node_data in nodes_list:
                raw_id = node_data.get("id")
                if not raw_id:
                    logger.warning(f"Skipping malformed node data in doc '{doc_id}': missing 'id'.")
                    continue
                
                # Construct unique namespaced ID
                namespaced_id = self._get_namespaced_id(doc_id, raw_id)
                node_type = node_data.get("type", "unknown")
                
                # Store node metadata
                node_attrs = {
                    "raw_id": raw_id,
                    "document_id": doc_id,
                    "type": node_type
                }
                # Capture any extra layout properties
                for k, v in node_data.items():
                    if k not in node_attrs:
                        node_attrs[k] = v
                        
                self.graph.add_node(namespaced_id, **node_attrs)
                self._doc_to_nodes[doc_id].add(namespaced_id)

            # 2. Parse and insert edges (links)
            links_list = serialized_graph_json.get("links", [])
            for link_data in links_list:
                source_raw = link_data.get("source")
                target_raw = link_data.get("target")
                relation = link_data.get("relation", "connected")
                
                if not source_raw or not target_raw:
                    logger.warning(f"Skipping malformed link in doc '{doc_id}': missing source or target.")
                    continue
                
                # Map source and target to their respective namespaced IDs
                source_namespaced = self._get_namespaced_id(doc_id, source_raw)
                target_namespaced = self._get_namespaced_id(doc_id, target_raw)
                
                # Verify nodes exist before establishing edges
                if not self.graph.has_node(source_namespaced) or not self.graph.has_node(target_namespaced):
                    logger.warning(
                        f"Skipping link {source_raw} -> {target_raw} in doc '{doc_id}': "
                        f"One or both nodes were not declared in the nodes registry."
                    )
                    continue
                
                edge_attrs = {"relation": relation}
                for k, v in link_data.items():
                    if k not in ["source", "target", "relation"]:
                        edge_attrs[k] = v
                        
                self.graph.add_edge(source_namespaced, target_namespaced, **edge_attrs)
                
            logger.info(
                f"Successfully merged graph for doc '{doc_id}'. "
                f"Active Master Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges."
            )
            
        except Exception as e:
            logger.error(f"Error loading graph for doc '{doc_id}': {str(e)}")
            raise RuntimeError(f"Failed to load document graph: {e}") from e

    def node_exists(self, node_id: str) -> bool:
        """
        Fast check to verify if a node exists in the master graph.
        
        Args:
            node_id: The node identifier (either namespaced or raw depending on query).
            
        Returns:
            True if the node is present in the graph.
        """
        return self.graph.has_node(node_id)

    def get_node_metadata(self, node_id: str) -> Dict[str, Any]:
        """
        Retrieves all attributes associated with a node.
        
        Args:
            node_id: The unique namespaced node ID.
            
        Returns:
            Dict containing node metadata attributes.
            
        Raises:
            KeyError: If node_id is not found in the graph.
        """
        if not self.node_exists(node_id):
            error_msg = f"Node '{node_id}' does not exist in the GraphStore."
            logger.error(error_msg)
            raise KeyError(error_msg)
        return dict(self.graph.nodes[node_id])

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Retrieves all directly connected neighbors (successors in the DiGraph).
        
        Args:
            node_id: The unique namespaced node ID.
            
        Returns:
            List of namespaced neighbor node IDs.
            
        Raises:
            KeyError: If node_id is not found in the graph.
        """
        if not self.node_exists(node_id):
            error_msg = f"Node '{node_id}' does not exist in the GraphStore."
            logger.error(error_msg)
            raise KeyError(error_msg)
        return list(self.graph.successors(node_id))

    def get_neighbors_by_relation(self, node_id: str, relation: str) -> List[str]:
        """
        Filters neighbors (successors) based on a specific relation type.
        
        Args:
            node_id: The unique namespaced node ID.
            relation: The target relation name (e.g. 'parent_of', 'captions').
            
        Returns:
            List of namespaced neighbor node IDs matching the relation.
            
        Raises:
            KeyError: If node_id is not found in the graph.
        """
        neighbors = self.get_neighbors(node_id)
        matching_neighbors = []
        
        for nbr in neighbors:
            edge_data = self.graph.get_edge_data(node_id, nbr)
            if edge_data and edge_data.get("relation") == relation:
                matching_neighbors.append(nbr)
                
        return matching_neighbors

    def get_document_nodes(self, document_id: str) -> List[str]:
        """
        Retrieves all node IDs belonging to a specific document.
        Executes in O(1) time complexity due to cached indexing.
        
        Args:
            document_id: The target document ID.
            
        Returns:
            List of namespaced node IDs belonging to the document.
        """
        return list(self._doc_to_nodes.get(document_id, set()))
