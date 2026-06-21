import json
import os
import sys

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

from src.context_retriever.main import DistributedContextRetriever
from src.context_retriever.database.qdrant_client import QdrantStorage
from src.integration.schema_adapter import adapt_pipeline_outputs

def main():
    print("=" * 60)
    print("  PERSON 5: INTEGRATION SCHEMA ADAPTER TEST")
    print("=" * 60)

    # 1. Load real artifacts
    elements_file = "output/quarterly_report_parsed.json"
    graph_file = "output/doc_graph.json"

    if not os.path.exists(elements_file) or not os.path.exists(graph_file):
        print(f"[ERROR] Required input files missing. Run run_p2.py first.")
        sys.exit(1)

    print(f"[*] Loading elements from: {elements_file}")
    with open(elements_file, "r", encoding="utf-8") as f:
        parsed_doc_data = json.load(f)
        parsed_elements = parsed_doc_data.get("elements", [])

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            doc_graph_json = json.load(f)
    except UnicodeDecodeError:
        with open(graph_file, "r", encoding="cp1252") as f:
            doc_graph_json = json.load(f)

    doc_id = "quarterly_report"

    # 2. Perform adaptation via formal module
    print("[*] Translating schemas using src.integration.schema_adapter...")
    adapted_elements, adapted_graph = adapt_pipeline_outputs(parsed_elements, doc_graph_json, doc_id)

    # 3. Validate adapted schemas against constraints
    print("\n" + "-" * 40)
    print("  SCHEMA VERIFICATION")
    print("-" * 40)
    
    # Verify first element
    first_elem = adapted_elements[0]
    required_elem_keys = ["element_id", "document_id", "page_number", "type", "content"]
    elem_ok = True
    for key in required_elem_keys:
        val = first_elem.get(key)
        if val is None:
            print(f"[FAIL] Element missing required key: {key}")
            elem_ok = False
        else:
            print(f"[OK] Element has {key} = {val}")
    
    # Verify graph root keys
    graph_keys_ok = "nodes" in adapted_graph and "links" in adapted_graph
    print(f"[{"OK" if graph_keys_ok else "FAIL"}] Graph has nodes and links keys")

    # Verify first node
    first_node = adapted_graph["nodes"][0]
    node_ok = "id" in first_node and first_node.get("id") is not None
    print(f"[{"OK" if node_ok else "FAIL"}] First node has 'id' = {first_node.get('id')}")

    # Verify first link
    first_link = adapted_graph["links"][0]
    link_ok = "source" in first_link and "target" in first_link and "relation" in first_link
    print(f"[{"OK" if link_ok else "FAIL"}] First link has source={first_link.get('source')}, target={first_link.get('target')}, relation={first_link.get('relation')}")

    if not (elem_ok and graph_keys_ok and node_ok and link_ok):
        print("\n[ERROR] Schema validation failed. Halting integration test.")
        sys.exit(1)

    print("\n[OK] Schema validation successful!")

    # 4. Initialize in-memory Qdrant retriever
    print("\n" + "-" * 40)
    print("  RETRIEVER INGESTION TEST")
    print("-" * 40)
    print("[*] Initializing DistributedContextRetriever with in-memory storage...")
    
    storage = QdrantStorage(url=":memory:")
    retriever = DistributedContextRetriever(qdrant_storage=storage)

    # 5. Ingest translated documents
    print("[*] Ingesting adapted elements and graph JSON...")
    stats = retriever.ingest_document(adapted_elements, adapted_graph)
    print(f"[OK] Ingest stats: {stats}")

    # 6. Run a quick query to test vector lookup + graph expansion
    print("\n" + "-" * 40)
    print("  RETRIEVAL TEST")
    print("-" * 40)
    query = "What was the company's revenue growth?"
    print(f"[*] Query: '{query}'")
    
    res = retriever.retrieve_context(query)
    print(f"[OK] Retrieval confidence: {res['confidence_score']:.4f}")
    print(f"[OK] Seeds found: {res['seed_count']}, Expanded nodes added: {res['expanded_count']}")
    print("\nRetrieved Markdown Context Preview:")
    print(res["context_bundle"]["formatted_markdown"][:500] + "\n...")

if __name__ == "__main__":
    main()
