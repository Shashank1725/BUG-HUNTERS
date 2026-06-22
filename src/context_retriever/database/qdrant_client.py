import logging
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue, 
    MatchAny,
    FilterSelector
)
from src.context_retriever.config import settings

logger = logging.getLogger(__name__)

class QdrantStorage:
    """
    Handles storage, ingestion, and vector search operations in Qdrant DB.
    Includes validation of vector dimensions, dynamic filtering for multi-document RAG,
    and deterministic ID mapping using UUID namespaces.
    """
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initializes the QdrantStorage client using configured settings.
        
        Args:
            url: Connection URL override. Defaults to settings.QDRANT_URL.
            api_key: Auth API key override. Defaults to settings.QDRANT_API_KEY.
        """
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        logger.info(f"Connecting to Qdrant service at endpoint: {self.url}")
        
        # try:
        #     self.client = QdrantClient(url=self.url, api_key=self.api_key)
        try:
            if self.url == ":memory:":
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key
                )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant vector database: {str(e)}")
            raise ConnectionError(f"Qdrant connection failed: {e}") from e

    def create_collection_if_not_exists(self, collection_name: Optional[str] = None) -> None:
        """
        Ensures a Qdrant collection is created with correct dimensions and distance metric.
        
        Args:
            collection_name: Target collection name override. Defaults to settings.COLLECTION_NAME.
        """
        name = collection_name or settings.COLLECTION_NAME
        dimension = settings.VECTOR_DIMENSION
        
        try:
            # Check if collection exists
            exists = self.client.collection_exists(name)
            if not exists:
                logger.info(f"Creating collection '{name}' with {dimension} dimensions (COSINE distance)...")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{name}' successfully created.")
            else:
                logger.debug(f"Collection '{name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating collection '{name}': {str(e)}")
            raise RuntimeError(f"Collection operations failed: {e}") from e

    def _generate_uuid_from_element(self, document_id: str, element_id: str) -> str:
        """
        Generates a deterministic UUID string based on document_id and element_id.
        Ensures Qdrant point IDs are unique across documents but repeatable for updates.
        
        Args:
            document_id: The document identifier.
            element_id: The element identifier.
            
        Returns:
            A UUID version 5 string representation.
        """
        # DNS Namespace is used as a standard seed namespace
        namespace_key = f"{document_id}_{element_id}"
        point_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, namespace_key)
        return str(point_uuid)

    def upsert_elements(
        self, 
        elements: List[Dict[str, Any]], 
        vectors: List[List[float]], 
        collection_name: Optional[str] = None
    ) -> None:
        """
        Uploads document layout elements and their embeddings to Qdrant.
        Validates elements and vector dimensions before ingestion.
        
        Args:
            elements: List of dictionaries matching the parsing schema (must include:
                      element_id, document_id, page_number, type, content).
            vectors: List of corresponding float vectors.
            collection_name: Target collection name override. Defaults to settings.COLLECTION_NAME.
            
        Raises:
            ValueError: On dimension or length mismatches.
            RuntimeError: On ingestion issues.
        """
        name = collection_name or settings.COLLECTION_NAME
        expected_dim = settings.VECTOR_DIMENSION
        
        if len(elements) != len(vectors):
            error_msg = f"Length mismatch: Elements ({len(elements)}) and Vectors ({len(vectors)}) must match."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        points = []
        for idx, (elem, vec) in enumerate(zip(elements, vectors)):
            # Validate vector dimensions
            if len(vec) != expected_dim:
                error_msg = (
                    f"Vector dimension mismatch at index {idx}! Vector size is {len(vec)}, "
                    f"but Qdrant index requires {expected_dim} dimensions."
                )
                logger.critical(error_msg)
                raise ValueError(error_msg)
            
            # Validate mandatory metadata fields
            required_fields = ["element_id", "document_id", "page_number", "type", "content"]
            for field in required_fields:
                if field not in elem:
                    error_msg = f"Mandatory field '{field}' is missing from element payload at index {idx}."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            # Generate stable point ID
            point_id = self._generate_uuid_from_element(elem["document_id"], elem["element_id"])
            
            # Construct payload matching Person 1 extraction model
            payload = {
                "element_id": elem["element_id"],
                "document_id": elem["document_id"],
                "page_number": int(elem["page_number"]),
                "type": str(elem["type"]),
                "content": str(elem["content"]),
            }
            
            # Preserve any auxiliary custom fields extracted (e.g. bounding boxes, levels, level descriptions)
            for k, v in elem.items():
                if k not in payload:
                    payload[k] = v
            
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        try:
            logger.info(f"Upserting {len(points)} elements to Qdrant collection '{name}'...")
            self.client.upsert(collection_name=name, points=points)
            logger.info("Upsert completed successfully.")
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant collection '{name}': {str(e)}")
            raise RuntimeError(f"Upsert operations failed: {e}") from e

    def search(
        self, 
        query_vector: List[float], 
        limit: int = 5, 
        document_ids: Optional[List[str]] = None, 
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries Qdrant using vector similarity. Supports single-document or multi-document
        filtering.
        
        Args:
            query_vector: Dense embedding vector representing the search query.
            limit: The maximum number of points to retrieve.
            document_ids: Optional list of document identifiers to restrict search scope.
            collection_name: Target collection name override. Defaults to settings.COLLECTION_NAME.
            
        Returns:
            A list of dictionary payloads, including relevance similarity scores.
        """
        name = collection_name or settings.COLLECTION_NAME
        
        # Build document filtering query if document scope is restricted
        search_filter = None
        if document_ids:
            if len(document_ids) == 1:
                search_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))]
                )
            else:
                search_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))]
                )

        try:
            logger.info(f"Searching collection '{name}' (Limit: {limit}, Filters: {bool(document_ids)})...")
            results = self.client.query_points(
                collection_name=name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit
        )
            
            # Map search response back to standard dict interfaces
            retrieved_hits = []
            for hit in results.points:
                if hit.payload is None:
                    continue
                hit_data = dict(hit.payload)
                hit_data["score"] = float(hit.score)
                retrieved_hits.append(hit_data)
                
            logger.info(f"Retrieved {len(retrieved_hits)} matching elements.")
            return retrieved_hits
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {str(e)}")
            raise RuntimeError(f"Search operations failed: {e}") from e

    def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> None:
        """
        Purges all element vectors associated with a specific document.
        
        Args:
            document_id: The document identifier to remove.
            collection_name: Target collection name override. Defaults to settings.COLLECTION_NAME.
        """
        name = collection_name or settings.COLLECTION_NAME
        
        try:
            logger.info(f"Deleting all vectors under document ID: '{document_id}' from collection '{name}'...")
            self.client.delete(
                collection_name=name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                    )
                )
            )
            logger.info("Deletion completed successfully.")
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}': {str(e)}")
            raise RuntimeError(f"Deletion operations failed: {e}") from e
