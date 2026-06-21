import logging
from typing import List, Dict, Optional, Any
from src.config import settings
from src.embeddings.generator import EmbeddingsGenerator
from src.database.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """
    Coordinates semantic vector retrieval by interfacing between EmbeddingsGenerator 
    and QdrantStorage. Handles query validation, embedding generation, 
    multi-document filtering, and retrieval candidate selection.
    """
    def __init__(
        self, 
        embeddings_generator: Optional[EmbeddingsGenerator] = None, 
        qdrant_storage: Optional[QdrantStorage] = None
    ):
        """
        Initializes the SemanticRetriever.
        
        Args:
            embeddings_generator: Optional instance override. If None, instantiates a default.
            qdrant_storage: Optional instance override. If None, instantiates a default.
        """
        self.generator = embeddings_generator or EmbeddingsGenerator()
        self.storage = qdrant_storage or QdrantStorage()
        logger.info("SemanticRetriever initialized successfully.")

    def retrieve_seeds(
        self, 
        query: str, 
        limit: Optional[int] = None, 
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a search query, generates embeddings, and retrieves matching elements
        from the vector database.
        
        Args:
            query: The user query string (must be non-empty and non-whitespace).
            limit: The maximum number of seed elements to retrieve (defaults to settings.TOP_K_RETRIEVAL).
            document_ids: Optional list of document identifiers to restrict search scope.
            
        Returns:
            A list of dictionary payloads matching the retrieval schema, containing:
            element_id, document_id, page_number, type, content, and score.
            
        Raises:
            ValueError: If the query is empty or malformed.
            RuntimeError: If embedding generation or vector search fails.
        """
        # Validate query parameter
        if not isinstance(query, str) or not query.strip():
            error_msg = "Search query must be a valid, non-empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)

        search_limit = limit if limit is not None else settings.TOP_K_RETRIEVAL
        logger.info(
            f"Retrieving seeds for query: '{query}' (Limit: {search_limit}, Document Filter: {document_ids})"
        )

        try:
            # 1. Generate query embedding vector
            query_vector = self.generator.get_embedding(query)
            
            # 2. Query Qdrant vector storage
            raw_hits = self.storage.search(
                query_vector=query_vector,
                limit=search_limit,
                document_ids=document_ids
            )
            
            # 3. Format results into output schema
            seeds = []
            for hit in raw_hits:
                seed_item = {
                    "element_id": hit.get("element_id"),
                    "document_id": hit.get("document_id"),
                    "page_number": hit.get("page_number"),
                    "type": hit.get("type"),
                    "content": hit.get("content"),
                    "score": hit.get("score", 0.0)
                }
                
                # Verify that all mandatory payload fields are present in the response mapping
                missing_fields = [k for k, v in seed_item.items() if v is None and k != "score"]
                if missing_fields:
                    logger.warning(
                        f"Retrieved hit missing mandatory fields: {missing_fields}. Hit payload: {hit}"
                    )
                    
                # Preserve any supplementary payload metadata fields (e.g. coordinates)
                for k, v in hit.items():
                    if k not in seed_item:
                        seed_item[k] = v
                        
                seeds.append(seed_item)
                
            logger.info(f"Successfully selected {len(seeds)} semantic seed elements.")
            return seeds
            
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Semantic retrieval operations failed: {str(e)}")
            raise RuntimeError(f"Semantic retrieval failed: {e}") from e
