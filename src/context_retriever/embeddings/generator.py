import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingsGenerator:
    """
    Generates semantic vector embeddings for text, tables, and visual descriptions.
    Uses SentenceTransformers under the hood. Implements lazy loading of the 
    underlying model and automatic dimension validation at runtime.
    """
    def __init__(self, model_name: Optional[str] = None, expected_dimension: Optional[int] = None):
        """
        Initializes the embeddings generator.
        
        Args:
            model_name: Optional model override. Defaults to settings.EMBEDDING_MODEL.
            expected_dimension: Optional dimension override. Defaults to settings.VECTOR_DIMENSION.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.expected_dimension = expected_dimension or settings.VECTOR_DIMENSION
        self._model: Optional[SentenceTransformer] = None

    # @property
    # def model(self) -> SentenceTransformer:
    #     """
    #     Property providing access to the SentenceTransformer model instance.
    #     Loads the model lazily on demand and performs dimension validation.
        
    #     Returns:
    #         The loaded SentenceTransformer instance.
            
    #     Raises:
    #         RuntimeError: If model initialization fails.
    #     """
    #     try:
    #         self._model = SentenceTransformer(self.model_name)
    #         logger.info("SentenceTransformer model successfully loaded.")
    #         self.validate_dimension()

    #     except ValueError:
    #         raise

    #     except Exception as e:
    #         logger.error(f"Failed to load embedding model '{self.model_name}': {str(e)}")
    #         raise RuntimeError(f"Embedding model initialization failed: {e}") from e
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(
                f"Lazily initializing SentenceTransformer model: '{self.model_name}'..."
            )

            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model successfully loaded.")
                self.validate_dimension()

            except ValueError:
                    raise

            except Exception as e:
                logger.error(
                    f"Failed to load embedding model '{self.model_name}': {str(e)}"
                )
                raise RuntimeError(
                    f"Embedding model initialization failed: {e}"
            ) from e

        return self._model
        
    def validate_dimension(self) -> bool:
        """
        Validates that the loaded SentenceTransformer model outputs vectors matching the 
        expected configuration dimension.
        
        Returns:
            True if the dimensions match.
            
        Raises:
            ValueError: If there is a mismatch between model output and expected settings.
            RuntimeError: If called before the model is loaded.
        """
        if self._model is None:
            raise RuntimeError("Cannot validate dimension: Model has not been initialized.")
        
        actual_dimension = self._model.get_embedding_dimension()
        if actual_dimension != self.expected_dimension:
            error_msg = (
                f"Embedding dimension mismatch! Model '{self.model_name}' outputs "
                f"{actual_dimension}-dimensional vectors, but the system configuration "
                f"expected {self.expected_dimension}-dimensional vectors."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Embedding dimension validation passed. Model outputs: {actual_dimension} dimensions.")
        return True

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single text input string.
        
        Args:
            text: The text to be converted into an embedding.
            
        Returns:
            A list of floats representing the embedding vector.
            
        Raises:
            ValueError: If input is not a non-empty string.
            RuntimeError: If embedding generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            error_msg = "Input text must be a valid, non-empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Invoking self.model property ensures lazy loading
            vector = self.model.encode(text, convert_to_numpy=True).tolist()
            return vector
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of text inputs.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of lists of floats representing embedding vectors.
            
        Raises:
            ValueError: If input is not a list of strings.
            RuntimeError: If batch embedding generation fails.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            error_msg = "Input must be a list of strings."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not texts:
            return []

        try:
            # Invoking self.model property ensures lazy loading
            vectors = self.model.encode(texts, convert_to_numpy=True).tolist()
            return vectors
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e
