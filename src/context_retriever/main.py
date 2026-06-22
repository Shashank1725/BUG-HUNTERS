import logging
from typing import List, Dict, Optional, Any

from src.context_retriever.config import settings
from src.context_retriever.embeddings.generator import EmbeddingsGenerator
from src.context_retriever.database.qdrant_client import QdrantStorage
from src.context_retriever.graph.networkx_store import GraphStore
from src.context_retriever.retrieval.semantic import SemanticRetriever
from src.context_retriever.retrieval.expansion import GraphExpander
from src.context_retriever.retrieval.confidence import ConfidenceScorer
from src.context_retriever.bundler.proof_chain import ProofChainBuilder
from src.context_retriever.bundler.bundle import ContextBundler

logger = logging.getLogger(__name__)

class DistributedContextRetriever:
    """
    The master orchestrator class for the Distributed Context Retrieval system.
    Wires together embeddings generation, Qdrant vector database storage, GraphStore
    relationship mappings, semantic search retrievers, graph context expanders, 
    confidence scorers, lineage proof builders, and context bundlers.
    """
    def __init__(
        self,
        embeddings_generator: Optional[EmbeddingsGenerator] = None,
        qdrant_storage: Optional[QdrantStorage] = None,
        graph_store: Optional[GraphStore] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
        graph_expander: Optional[GraphExpander] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        proof_chain_builder: Optional[ProofChainBuilder] = None,
        context_bundler: Optional[ContextBundler] = None
    ):
        """
        Initializes the DistributedContextRetriever with dependency injection support.
        """
        # Wire core infrastructure components
        self.generator = embeddings_generator or EmbeddingsGenerator()
        self.storage = qdrant_storage or QdrantStorage()
        self.graph_store = graph_store or GraphStore()

        # Wire retrieval engine components
        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            embeddings_generator=self.generator,
            qdrant_storage=self.storage
        )
        self.graph_expander = graph_expander or GraphExpander(
            graph_store=self.graph_store
        )

        # Wire evaluation and bundling components
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.proof_chain_builder = proof_chain_builder or ProofChainBuilder()
        self.context_bundler = context_bundler or ContextBundler()

        logger.info("DistributedContextRetriever orchestrator successfully initialized.")

    def ingest_document(self, elements: List[Dict[str, Any]], graph_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a parsed multi-modal document layout:
        1. Ensures the target collection is set up.
        2. Generates semantic vector embeddings for all layout nodes in batch.
        3. Upserts elements and vectors into the Qdrant DB.
        4. Merges relational edge properties into the GraphStore.
        
        Args:
            elements: List of elements from Person 1 (must include element_id, 
                      document_id, page_number, type, and content).
            graph_json: Relational graph link representation from Person 2.
            
        Returns:
            Dict containing ingestion stats (elements_ingested, graph_nodes, graph_edges).
        """
        if not elements:
            raise ValueError("Elements list cannot be empty for ingestion.")

        # Extract document ID from contents to validate context boundaries
        document_id = elements[0].get("document_id") or graph_json.get("graph", {}).get("document_id")
        if not document_id:
            raise ValueError("Document ID is missing from ingestion payloads.")

        logger.info(f"Starting pipeline ingestion for document ID: '{document_id}'...")

        try:
            # 1. Verify vector database collection structure is active
            self.storage.create_collection_if_not_exists()

            # 2. Extract contents and run batch embedding conversion
            contents = [str(elem.get("content", "")) for elem in elements]
            logger.info(f"Generating vectors for {len(elements)} layout elements...")
            vectors = self.generator.get_embeddings_batch(contents)

            # 3. Upsert data to Qdrant Storage
            self.storage.upsert_elements(elements=elements, vectors=vectors)

            # 4. Enrich graph nodes with metadata fields from elements list
            # Since graph JSON nodes might lack content, page numbers, or bounding boxes,
            # we merge those attributes from the parser elements list to ensure completeness.
            elements_map = {elem["element_id"]: elem for elem in elements}
            enriched_nodes = []
            for node in graph_json.get("nodes", []):
                node_id = node.get("id")
                enriched_node = dict(node)
                if node_id in elements_map:
                    for k, v in elements_map[node_id].items():
                        if k != "element_id":  # keep GraphStore id conventions
                            enriched_node[k] = v
                enriched_nodes.append(enriched_node)
            
            enriched_graph_json = dict(graph_json)
            enriched_graph_json["nodes"] = enriched_nodes

            # Load edge relationship mappings into the master GraphStore
            self.graph_store.load_document_graph(
                serialized_graph_json=enriched_graph_json, 
                document_id=document_id
            )

            # 5. Extract statistics for confirmation
            doc_nodes = self.graph_store.get_document_nodes(document_id)
            doc_edges_count = 0
            
            # Count edges associated with this document specifically
            for u, v in self.graph_store.graph.edges:
                node_data = self.graph_store.graph.nodes[u]
                if node_data.get("document_id") == document_id:
                    doc_edges_count += 1

            stats = {
                "elements_ingested": len(elements),
                "graph_nodes": len(doc_nodes),
                "graph_edges": doc_edges_count
            }
            logger.info(f"Ingestion completed for document '{document_id}': {stats}")
            return stats

        except Exception as e:
            logger.error(f"Ingestion failed for document '{document_id}': {str(e)}")
            raise RuntimeError(f"Ingestion pipeline execution failed: {e}") from e

    def retrieve_context(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None, 
        token_limit: int = 4000
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end multi-modal context retrieval query pipeline:
        1. Queries Qdrant using dense vector lookup to find seed nodes.
        2. Expands retrieved seeds using document graph structures in GraphStore.
        3. Calculates a combined similarity/neighborhood density confidence score.
        4. Generates an explainable proof chain mapping derivations back to seeds.
        5. Packages context elements into Markdown, restricted by token budgets.
        
        Args:
            query: The user query string (non-empty).
            document_ids: Optional list of document ID filters to scope search results.
            token_limit: The maximum word token budget for prompt synthesis.
            
        Returns:
            Dict mapping: query, confidence_score, context_bundle, retrieved_elements,
            seed_count, and expanded_count.
        """
        logger.info(f"Initiating context retrieval pipeline for query: '{query}'...")

        try:
            # Step 1: Semantic seed retrieval
            seeds = self.semantic_retriever.retrieve_seeds(
                query=query, 
                limit=settings.TOP_K_RETRIEVAL, 
                document_ids=document_ids
            )
            
            # Step 2: Controlled layout relationship graph expansion
            # Returns a unified, deduplicated list containing both seeds and expansions
            all_elements = self.graph_expander.expand_context(seeds)
            expanded_only = [el for el in all_elements if el.get("expansion_reason") != "seed"]

            # Step 3: Compute retrieval confidence scoring metric
            confidence = self.confidence_scorer.score_retrieval(
                seeds=seeds, 
                expanded_nodes=all_elements
            )

            # Step 4: Construct explainable proof lineage mappings
            proof_chain = self.proof_chain_builder.generate_lineage(
                seeds=seeds, 
                expanded_nodes=all_elements
            )

            # Step 5: Format context Markdown bundle and apply token budgets
            bundle = self.context_bundler.build_context_bundle(
                elements=all_elements, 
                proof_chain=proof_chain, 
                token_limit=token_limit
            )

            result = {
                "query": query,
                "confidence_score": confidence,
                "context_bundle": bundle,  # contains: formatted_markdown, token_count, citation_metadata
                "retrieved_elements": bundle["citation_metadata"],
                "seed_count": len(seeds),
                "expanded_count": len(expanded_only)
            }
            logger.info(
                f"Context retrieval finished. Seeds: {len(seeds)}, Expanded: {len(expanded_only)}, "
                f"Confidence Score: {confidence:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Context retrieval pipeline failed: {str(e)}")
            raise RuntimeError(f"Retrieval pipeline execution failed: {e}") from e
