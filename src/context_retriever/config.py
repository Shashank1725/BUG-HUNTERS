from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Distributed Context Retrieval System.
    Inherits from pydantic_settings.BaseSettings to dynamically load settings
    from environment variables (e.g. QDRANT_URL) or a local .env file.
    """
    
    # Qdrant Database Settings
    QDRANT_URL: str = Field(
        default="http://localhost:6333", 
        description="The connection endpoint of the Qdrant vector database."
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default=None, 
        description="API security key to authenticate Qdrant requests."
    )
    COLLECTION_NAME: str = Field(
        default="document_elements", 
        description="The name of the target vector collection in Qdrant."
    )
    
    # Embedding Configuration Settings
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5", 
        description="The Sentence-Transformers model used to generate element embeddings."
    )
    VECTOR_DIMENSION: int = Field(
        default=384, 
        description="Dimensions of the vector embeddings (384 for BAAI/bge-small-en-v1.5)."
    )
    
    # Retrieval Configuration Settings
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.4, 
        description="Minimum similarity score threshold for dynamic retrieval confidence."
    )
    TOP_K_RETRIEVAL: int = Field(
        default=5, 
        description="The number of seed elements to pull from Qdrant vector storage."
    )
    GRAPH_EXPANSION_DEPTH: int = Field(
        default=1, 
        description="The relational traversal depth (in number of node links) for context expansion."
    )
    
    # Reranking and Final Selection Config
    RERANK_TOP_K: int = Field(
        default=15,
        description="The intermediate number of retrieved documents to send to the reranker."
    )
    FINAL_CONTEXT_K: int = Field(
        default=5,
        description="The final number of context chunks returned to the generator."
    )

    # Enable loading variables from environment and local .env files
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

# Global settings instance to import configurations across modules
settings = Settings()
