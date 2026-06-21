import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContextBundler:
    """
    Assembles, formats, and restricts retrieved document layout elements 
    into a structured Markdown context bundle for response synthesis.
    """
    def build_context_bundle(
        self, 
        elements: List[Dict[str, Any]], 
        proof_chain: Dict[str, str], 
        token_limit: int = 4000
    ) -> Dict[str, Any]:
        """
        Assembles, deduplicates, and sorts layout elements into a structured Markdown bundle,
        respecting token restrictions and decorating elements with lineage metadata.
        
        Args:
            elements: Combined list of seed and expanded element dictionaries.
            proof_chain: Lineage mapping from ProofChainBuilder.
            token_limit: The maximum word token limit budget (defaults to 4000).
            
        Returns:
            A dictionary containing:
            - formatted_markdown: Unified string representing context bundle.
            - token_count: Total word token count of formatted_markdown.
            - citation_metadata: List of elements successfully packed within the limit.
        """
        if not elements:
            logger.info("No elements provided to bundler. Returning empty package.")
            return {
                "formatted_markdown": "# Retrieved Context\n\nNo relevant context retrieved.",
                "token_count": 5,
                "citation_metadata": []
            }

        try:
            # 1. Deduplicate elements based on globally unique namespaced key
            deduplicated_map: Dict[str, Dict[str, Any]] = {}
            for elem in elements:
                doc_id = elem.get("document_id")
                elem_id = elem.get("element_id")
                if not doc_id or not elem_id:
                    continue
                namespaced_key = f"{doc_id}::{elem_id}"
                # If element already exists, keep seed version or first encounter
                if namespaced_key not in deduplicated_map:
                    deduplicated_map[namespaced_key] = elem
                    
            sorted_elements = list(deduplicated_map.values())

            # 2. Sort elements by document_id -> page_number -> element_id
            # Natural layout sequencing helps LLMs read sections sequentially
            sorted_elements.sort(
                key=lambda x: (
                    str(x.get("document_id", "")),
                    int(x.get("page_number", 1)),
                    str(x.get("element_id", ""))
                )
            )

            # 3. Build Markdown structure, enforcing token limits incrementally
            accumulated_markdown = "# Retrieved Context\n"
            citation_metadata = []
            
            current_doc_id = None
            current_page_number = None

            for elem in sorted_elements:
                elem_doc_id = elem["document_id"]
                elem_page_num = elem["page_number"]
                elem_id = elem["element_id"]
                
                # Check if headers are required
                doc_header_needed = (elem_doc_id != current_doc_id)
                page_header_needed = doc_header_needed or (elem_page_num != current_page_number)
                
                # Construct candidate segment string
                segment = ""
                if doc_header_needed:
                    segment += f"\n## Document: {elem_doc_id}\n"
                if page_header_needed:
                    segment += f"\n### Page: {elem_page_num}\n"
                    
                elem_type_str = str(elem.get("type", "Element")).capitalize()
                namespaced_key = f"{elem_doc_id}::{elem_id}"
                lineage_info = proof_chain.get(namespaced_key, "[Unknown Lineage]")
                
                content = str(elem.get("content", ""))
                segment += f"\n#### [{elem_type_str}] (Source: {lineage_info})\n{content}\n"

                # Estimate token size of the bundle including the new segment
                candidate_markdown = accumulated_markdown + segment
                candidate_token_count = len(candidate_markdown.split())

                if candidate_token_count <= token_limit:
                    accumulated_markdown = candidate_markdown
                    citation_metadata.append({
                        "element_id": elem_id,
                        "document_id": elem_doc_id,
                        "page_number": elem_page_num,
                        "type": elem.get("type"),
                        "content": content,
                        "score": float(elem.get("score", 0.0)),
                        "lineage": lineage_info
                    })
                    
                    # Update active grouping headers
                    if doc_header_needed:
                        current_doc_id = elem_doc_id
                    if page_header_needed:
                        current_page_number = elem_page_num
                else:
                    # Token budget exhausted, halt packing
                    logger.warning(
                        f"Context bundler limit reached! Truncated at {len(citation_metadata)} "
                        f"elements (estimated tokens: {len(accumulated_markdown.split())})."
                    )
                    break

            final_token_count = len(accumulated_markdown.split())
            logger.info(
                f"Successfully built context bundle. packed elements: {len(citation_metadata)}, "
                f"tokens: {final_token_count}."
            )
            return {
                "formatted_markdown": accumulated_markdown,
                "token_count": final_token_count,
                "citation_metadata": citation_metadata
            }

        except Exception as e:
            logger.error(f"Failed to build context bundle: {str(e)}")
            raise RuntimeError(f"Context bundle compilation failed: {e}") from e
