import os
import sys
import json
import argparse
import google.generativeai as genai

# Ensure top-level src is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline import parse_document
from p2_pipeline import GraphPipeline, PipelineConfig
from src.integration.schema_adapter import adapt_pipeline_outputs
from src.context_retriever.main import DistributedContextRetriever
from src.context_retriever.database.qdrant_client import QdrantStorage

def run_qa_synthesis(query: str, context_bundle: dict) -> str:
    """
    Synthesizes the grounded answer using Gemini or a rule-based fallback.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    markdown_context = context_bundle.get("formatted_markdown", "")
    citations = context_bundle.get("citation_metadata", [])

    if api_key:
        print("[*] Contacting Gemini API for grounded answer synthesis...")
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""You are an expert financial analyst.
Answer the user's question based strictly on the provided document context.
If the context doesn't contain the answer, say "I cannot find the answer in the provided context."

CRITICAL REQUIREMENT: Every claim or fact you extract must be cited inline using the format: [document_id::element_id].
For example: "Revenue was $10M [report::report_p1_el4] and gross margins rose 18% [report::report_p1_el12]."

Retrieved Context:
{markdown_context}

Question:
{query}

Answer:"""
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[!] Gemini API call failed: {e}. Falling back to rule-based generation.")
    
    # Rule-based fallback if API key is not present or failed
    print("[*] Generating rule-based citation response (No GEMINI_API_KEY in env)...")
    
    # Search for matching citation sources in context
    matched_citations = []
    response_lines = ["Based on the retrieved context:"]
    
    for c in citations:
        source_id = f"{c['document_id']}::{c['element_id']}"
        content_snippet = c['content'].strip().replace("\n", " ")
        if len(content_snippet) > 100:
            content_snippet = content_snippet[:100] + "..."
            
        if c['type'] == 'heading':
            response_lines.append(f"- Section '{content_snippet}' was identified as a structural header [{source_id}].")
        elif c['type'] == 'caption':
            response_lines.append(f"- The visual element was described by the caption '{content_snippet}' [{source_id}].")
        else:
            response_lines.append(f"- Document content states: \"{content_snippet}\" [{source_id}].")
            
    response_lines.append("\nNote: Set GEMINI_API_KEY in your environment to generate a fully synthesized response.")
    return "\n".join(response_lines)

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Semantic Integration Demo Flow (Person 5)")
    parser.add_argument("pdf_path", help="Path to the PDF document to parse")
    parser.add_argument("--query", default="What was the company's revenue growth?", help="Question to ask the document")
    parser.add_argument("--output_dir", default="./output", help="Output directory for intermediate files")
    args = parser.parse_args()

    print("=" * 70)
    print("  DELL FUTUREMINDS AI HACKATHON — MULTI-MODAL PIPELINE DEMO")
    print("=" * 70)

    doc_id = os.path.splitext(os.path.basename(args.pdf_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Person 1 - Parse Document
    print(f"\n[STEP 1] Ingesting & Parsing PDF: {args.pdf_path}")
    print("-" * 60)
    parsed_doc = parse_document(args.pdf_path, output_dir=args.output_dir)
    print(f"[✓] Extracted {len(parsed_doc.elements)} structured elements.")

    # 2. Person 2 - Build Relationship Graph
    print(f"\n[STEP 2] Building Layout Relationship Graph")
    print("-" * 60)
    config = PipelineConfig(
        output_dir=args.output_dir,
        use_ml_similarity=True,
        use_entity_cooccurrence=False,
    )
    graph_builder = GraphPipeline(config)
    doc_graph = graph_builder.run_from_doc(parsed_doc)
    
    # Read generated graph JSON
    graph_json_path = os.path.join(args.output_dir, "doc_graph.json")
    with open(graph_json_path, "r", encoding="utf-8") as f:
        doc_graph_json = json.load(f)

    # 3. Person 5 - Adapt schemas
    print(f"\n[STEP 3] Running Person 5 Schema Adapter")
    print("-" * 60)
    raw_elements = [e.to_dict() for e in parsed_doc.elements]
    adapted_elements, adapted_graph = adapt_pipeline_outputs(raw_elements, doc_graph_json, doc_id)
    print(f"[✓] Translated element properties and multi-graph nodes/links successfully.")

    # 4. Person 3 - Ingest into Retriever
    print(f"\n[STEP 4] Ingesting to Distributed Vector & Graph Store")
    print("-" * 60)
    storage = QdrantStorage(url=":memory:")
    retriever = DistributedContextRetriever(qdrant_storage=storage)
    stats = retriever.ingest_document(adapted_elements, adapted_graph)
    print(f"[✓] Loaded {stats['elements_ingested']} elements into Vector DB.")
    print(f"[✓] Loaded {stats['graph_nodes']} nodes and {stats['graph_edges']} edges into NetworkX Store.")

    # 5. Person 3 - Context Retrieval
    print(f"\n[STEP 5] Executing Context Retrieval")
    print("-" * 60)
    print(f"Query: '{args.query}'")
    retrieval_res = retriever.retrieve_context(args.query)
    print(f"[✓] Seeds retrieved: {retrieval_res['seed_count']}, Expanded nodes added: {retrieval_res['expanded_count']}")
    print(f"[✓] Total retrieval confidence: {retrieval_res['confidence_score']:.4f}")

    # 6. Person 4 - QA Synthesis
    print(f"\n[STEP 6] QA Synthesis & Citations")
    print("-" * 60)
    answer = run_qa_synthesis(args.query, retrieval_res["context_bundle"])
    print("\nFINAL GROUNDED ANSWER:")
    print(answer)
    print("=" * 70)

if __name__ == "__main__":
    main()
