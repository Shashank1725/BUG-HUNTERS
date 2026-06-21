import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def adapt_elements(parsed_elements: List[Dict[str, Any]], document_id: str) -> List[Dict[str, Any]]:
    """
    Translates Person 1's elements schema into the schema expected by Person 3.
    
    Mapping changes:
    - Adds 'document_id' (mandatory for Person 3 vector database and grouping).
    - Maps 'page' to 'page_number' (Person 3 uses 'page_number' for page sorting).
    """
    adapted_elements = []
    for idx, elem in enumerate(parsed_elements):
        elem_dict = dict(elem)
        
        # Verify required base fields from Person 1
        if "element_id" not in elem_dict or "content" not in elem_dict:
            logger.warning(f"Element at index {idx} is missing base fields (element_id/content). Skipping.")
            continue
            
        # Ingest document_id scope
        elem_dict["document_id"] = document_id
        
        # Map page fields
        elem_dict["page_number"] = elem_dict.get("page", 1)
        
        adapted_elements.append(elem_dict)
        
    logger.info(f"Schema Adapter: Successfully adapted {len(adapted_elements)} elements for document '{document_id}'.")
    return adapted_elements

def adapt_graph(doc_graph_json: Dict[str, Any], document_id: str) -> Dict[str, Any]:
    """
    Translates Person 2's relationship graph schema into the NetworkX schema expected by Person 3.
    
    Mapping changes:
    - Sets graph level metadata 'graph': {'document_id': document_id}
    - Nodes: Maps 'element_id' to 'id'.
    - Edges -> Links: Maps key 'edges' to 'links', and key 'edge_type' to 'relation'.
    """
    adapted_graph = {
        "graph": {
            "document_id": document_id
        },
        "nodes": [],
        "links": []
    }

    # 1. Map nodes
    for node in doc_graph_json.get("nodes", []):
        adapted_node = dict(node)
        
        # Person 3 expects 'id' as the unique node identifier
        if "element_id" in node:
            adapted_node["id"] = node["element_id"]
        elif "id" not in node:
            logger.warning(f"Skipping malformed graph node: missing node identifier.")
            continue
            
        adapted_graph["nodes"].append(adapted_node)

    # 2. Map edges to links
    for edge in doc_graph_json.get("edges", []):
        if "source" not in edge or "target" not in edge:
            logger.warning(f"Skipping malformed edge: missing source/target.")
            continue
            
        # Person 3 expects 'links' list with 'relation' describing edge type
        adapted_link = {
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge.get("edge_type", "connected"),
            "weight": edge.get("weight", 1.0)
        }
        
        # Preserve any auxiliary edge metadata
        for k, v in edge.items():
            if k not in ["source", "target", "edge_type", "weight"]:
                adapted_link[k] = v
                
        adapted_graph["links"].append(adapted_link)

    logger.info(
        f"Schema Adapter: Translated graph nodes ({len(adapted_graph['nodes'])}) "
        f"and links ({len(adapted_graph['links'])}) for document '{document_id}'."
    )
    return adapted_graph

def adapt_pipeline_outputs(
    parsed_elements: List[Dict[str, Any]], 
    doc_graph_json: Dict[str, Any], 
    document_id: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience wrapper to translate both elements list and graph JSON in a single call.
    """
    adapted_elements = adapt_elements(parsed_elements, document_id)
    adapted_graph = adapt_graph(doc_graph_json, document_id)
    return adapted_elements, adapted_graph
