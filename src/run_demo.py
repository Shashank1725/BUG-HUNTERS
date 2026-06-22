import os
import sys
import json
import argparse
import google.generativeai as genai

# Ensure both the repository root and 'src' directory are in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
repo_root = os.path.dirname(current_dir)                 # root/

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import io
# Intercept stdout/stderr to prevent UnicodeEncodeError on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pipeline import parse_document
from p2_pipeline import GraphPipeline, PipelineConfig
from src.integration.schema_adapter import adapt_pipeline_outputs
from src.context_retriever.main import DistributedContextRetriever
from src.context_retriever.database.qdrant_client import QdrantStorage
from qa_engine import QAEngine, ContextElement

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Semantic Integration Demo Flow (Person 5)")
    parser.add_argument("pdf_path", help="Path to the PDF document to parse")
    parser.add_argument("--query", default="What was the company's revenue growth?", help="Question to ask the document")
    parser.add_argument("--output_dir", default="./output", help="Output directory for intermediate files")
    args = parser.parse_args()

    # Check for required GROQ_API_KEY in environment to fail fast with a clean error message
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY not found in environment.")
        sys.exit(1)

    print("=" * 70)
    print("  DELL FUTUREMINDS AI HACKATHON - MULTI-MODAL PIPELINE DEMO")
    print("=" * 70)

    doc_id = os.path.splitext(os.path.basename(args.pdf_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Person 1 - Parse Document
    print(f"\n[STEP 1] Ingesting & Parsing PDF: {args.pdf_path}")
    print("-" * 60)
    parsed_doc = parse_document(args.pdf_path, output_dir=args.output_dir)
    print(f"[OK] Extracted {len(parsed_doc.elements)} structured elements.")

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
    try:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            doc_graph_json = json.load(f)
    except UnicodeDecodeError:
        with open(graph_json_path, "r", encoding="cp1252") as f:
            doc_graph_json = json.load(f)

    # 3. Person 5 - Adapt schemas
    print(f"\n[STEP 3] Running Person 5 Schema Adapter")
    print("-" * 60)
    raw_elements = [e.to_dict() for e in parsed_doc.elements]
    adapted_elements, adapted_graph = adapt_pipeline_outputs(raw_elements, doc_graph_json, doc_id)
    print(f"[OK] Translated element properties and multi-graph nodes/links successfully.")

    # 4. Person 3 - Ingest into Retriever
    print(f"\n[STEP 4] Ingesting to Distributed Vector & Graph Store")
    print("-" * 60)
    storage = QdrantStorage(url=":memory:")
    retriever = DistributedContextRetriever(qdrant_storage=storage)
    stats = retriever.ingest_document(adapted_elements, adapted_graph)
    print(f"[OK] Loaded {stats['elements_ingested']} elements into Vector DB.")
    print(f"[OK] Loaded {stats['graph_nodes']} nodes and {stats['graph_edges']} edges into NetworkX Store.")

    # 5. Person 3 - Context Retrieval
    print(f"\n[STEP 5] Executing Context Retrieval")
    print("-" * 60)
    print(f"Query: '{args.query}'")
    retrieval_res = retriever.retrieve_context(args.query)
    print(f"[OK] Seeds retrieved: {retrieval_res['seed_count']}, Expanded nodes added: {retrieval_res['expanded_count']}")
    print(f"[OK] Total retrieval confidence: {retrieval_res['confidence_score']:.4f}")

    # 6. Person 4 - QA Synthesis (Integrated QAEngine)
    print(f"\n[STEP 6] QA Synthesis & Citations via QAEngine")
    print("-" * 60)
    
    # Convert retrieved elements dicts to ContextElement dataclass instances
    context_elements = [
        ContextElement(
            element_id=el["element_id"],
            type=el["type"],
            content=el["content"],
            page=el.get("page_number") or el.get("page", 1),
            doc_id=el.get("document_id") or el.get("doc_id", ""),
            relevance_score=el.get("score") or el.get("relevance_score", 1.0)
        )
        for el in retrieval_res["retrieved_elements"]
    ]

    qa_engine = QAEngine()
    
    qa_result = qa_engine.ask(
        question=args.query,
        context_elements=context_elements
    )

    print(qa_result.answer)
    print(qa_result.confidence)
    print(qa_result.citations)

if __name__ == "__main__":
    main()
