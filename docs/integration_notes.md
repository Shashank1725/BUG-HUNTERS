# Integration Notes — Person 5 (Integration & Evaluation)

This document summarizes the integration layer, schema alignments, observed behaviors, and demo instructions for the Multi-Modal Semantic Integration pipeline.

---

## 1. Schema Integration Layer (Adapter)

To connect Person 1 (Parsing) and Person 2 (Graph Builder) with Person 3 (Context Retriever) without modifying validated upstream code, we created a schema adapter located at `src/integration/schema_adapter.py`.

### Elements Schema Translation
* **Upstream (Person 1):** Outputs elements with the key `"page"`. Does not contain a `"document_id"` key.
* **Downstream (Person 3):** Requires `"page_number"` and `"document_id"` on each element to execute vector storage and layout sorting.
* **Resolution:** The adapter maps `"page_number"` = `"page"` and injects the active `"document_id"` string.

### Graph Schema Translation
* **Upstream (Person 2):** Exports a custom JSON with `"edges"` (using `"edge_type"`) and nodes (using `"element_id"`).
* **Downstream (Person 3):** Expects NetworkX node-link JSON with `"links"` (using `"relation"` as key) and nodes (using `"id"` as key).
* **Resolution:** The adapter dynamically translates the JSON keys:
  * Renames `"edges"` list to `"links"`.
  * Renames edge `"edge_type"` to `"relation"`.
  * Renames node `"element_id"` to `"id"`.
  * Injects `"graph": {"document_id": document_id}` metadata.

---

## 2. Structural Observation: Graph Edge Collapsing

During integration testing, a structural discrepancy was identified between the graph construction output and the retrieval storage input:
* **Person 2's Output:** `149` edges.
* **Person 3's Ingestion:** `85` edges.

### Why this happens:
* Person 2 uses a `networkx.MultiDiGraph` representation, which allows multiple parallel edges between the same two nodes (e.g. Node A is both `proximity` and `same_section` to Node B).
* Person 3's `GraphStore` uses a standard `networkx.DiGraph` representation, which allows only a single edge between any two nodes.
* During ingestion, parallel edges are collapsed into a single edge, with the last edge parsed overwriting previous relationship types.

### Impact on Retrieval:
* **Low/Minimal:** The dynamic graph expansion heuristics successfully traversed key relationship paths (like caption-table links) during benchmark query testing.
* **Long-Term Recommendation:** Update `src/context_retriever/graph/networkx_store.py` to use `nx.MultiDiGraph` to match Person 2's fidelity, or modify the adapter to aggregate multiple relation types into a list attribute.

---

## 3. How to Run the Demo

The pipeline can be executed end-to-end using the master demo runner.

### Environment Requirements
Ensure the following packages are installed:
```bash
py -3.13 -m pip install qdrant-client pydantic-settings groq
```

### Execution Command
Run the following command from the repository root:
```bash
$env:PYTHONPATH=".;src"; $env:PYTHONUTF8="1"; py -3.13 src/run_demo.py sample_docs/quarterly_report.pdf
```

### Answer Synthesis Mode
* **Grounded LLM (Required):** Set your API key in the environment to generate a fully synthesized response via Groq:
  ```bash
  $env:GROQ_API_KEY="gsk_..."
  ```
