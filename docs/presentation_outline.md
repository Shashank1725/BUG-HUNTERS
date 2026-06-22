# Presentation Outline: Multi-Modal Semantic Integration for Document Understanding

**Team Name:** BUG-HUNTERS  
**Target Category:** Dell FutureMinds AI Hackathon — Bangalore  
**Presentation Time:** 5–7 Minutes  

---

## Slide 1: Title Slide (First Impression)
* **Title:** Multi-Modal Semantic Integration for Intelligent Unstructured Document Understanding
* **Subtitle:** Moving Beyond Flat-Chunk Naive RAG to Layout-Aware Document Intelligence
* **Team BUG-HUNTERS:**
  * Person 1: Document Ingestion & Parsing
  * Person 2: Relationship Modeling & Graph Construction
  * Person 3: Distributed Context Retrieval
  * Person 4: QA Synthesis & Generation
  * Person 5: Integration, Evaluation, and Demo Presentation
* **Visual Concept:** Harmonics dark-mode background showing a document morphing into a connected network graph.

---

## Slide 2: The Pitfalls of Naive RAG (The Problem)
* **What Naive RAG Gets Wrong:**
  * **Flat Chunking:** Chops tables, headers, and text paragraphs arbitrarily, breaking structural meaning.
  * **Keyword / Vector Blindness:** Fails to link an image to its describing caption if they are separated.
  * **Contextual Isolation:** Treats sections of a report as independent blocks rather than parts of a structured hierarchy.
* **Judges' Hook (Dell Statement):** "Keyword search is dead. Naive RAG is insufficient. Multi-modal company knowledge is distributed across paragraphs, tables, images, and captions."
* **Our Solution Goal:** Build an AI analyst that *understands relationship networks*, *remembers layout structures*, and *recovers distributed context*.

---

## Slide 3: System Architecture (How it Works)
* **The Structured Pipeline:**
  ```text
  Raw PDF / Word / Excel
         ↓ [Person 1: Parsed layout elements + auto-captioning]
  Structured Elements
         ↓ [Person 2: Relationship heuristics + ML co-reference]
  Multi-Modal Document Graph
         ↓ [Person 5: Schema Adapter integration layer]
  Ingestion Layer
         ↓ [Person 3: Dense vector storage (Qdrant) + Graph Store]
  Distributed Context Retrieval
         ↓ [Person 4: Answer synthesis with lineage tracking]
  Grounded Answer + Citations
  ```
* **Core Technical Pillars:**
  1. Multi-modal layout parsing.
  2. Directed relationship graph modeling.
  3. Hybrid vector-graph retrieval.

---

## Slide 4: Ingestion & Multi-Modal Parsing (Person 1)
* **Features:**
  * Formats: PDF, DOCX, XLSX, CSV, PNG, JPG.
  * Content Extraction: Extracts text blocks, lists, headings, and tables as clean Markdown.
  * Visual Element Processing: Automatic extraction of figures and image files.
  * Local Vision-AI: Integrated with `BLIP-image-captioning` model locally (CPU/GPU) to generate high-accuracy semantic captions for figures.
* **Output:** Normalizes all document data into a unified, layout-aware JSON schema containing coordinates and parent heading IDs.

---

## Slide 5: Layout Relationship Graph Construction (Person 2)
* **Modeling the Document Network:**
  * Nodes represent individual parsed elements (paragraphs, tables, images, captions).
  * Edges represent logical layout and semantic relationships.
* **Relationship Types Extracted:**
  * `sequential`: Reading order of elements.
  * `parent_child`: Layout hierarchy (headings to content).
  * `same_section`: Siblings sharing a parent heading.
  * `caption_of`: Linking figures and tables to their labels.
  * `co_reference`: Semantic links established via high-similarity sentence embeddings.
* **Visuals:** Showcase the exported NetworkX graph visualizers (`graph_full.png` and `graph_by_type.png`).

---

## Slide 6: The Integration Challenge — Schema Adapter (Person 5)
* **The Reality of Multi-Developer Pipelines:**
  * Upstream and downstream components are developed independently, creating schema mismatches.
* **Mismatches Solved:**
  * **Element Keys:** Translated Person 1's `page` to Person 3's required `page_number` and injected missing document scope metadata (`document_id`).
  * **Graph Model:** Translated Person 2's custom MultiDiGraph (`edges`, `element_id`, `edge_type`) to Person 3's required NetworkX schema (`links`, `id`, `relation`).
* **Impact:** The integration adapter acts as a buffer layer, successfully linking the parsed documents to the database without destabilizing validated upstream logic.

---

## Slide 7: Distributed Context Retrieval (Person 3)
* **Moving Beyond Vector Search:**
  * **Step 1: Seed Lookup:** Performs a dense semantic search in Qdrant (in-memory) to locate the top relevant entry points.
  * **Step 2: Dynamic Graph Expansion:** Traverses the NetworkX relationship graph around the seed elements to capture missing but logically connected context (e.g. adjacent reading flow, parent headers, linked captions/tables).
  * **Step 3: Confidence & Lineage Scorer:** Computes an audit trail showing the path from each retrieved element back to its seed entry point.
  * **Step 4: Token-Budgeted Formatting:** Packages elements into a structured Markdown bundle within a strict LLM token limit.

---

## Slide 8: Answer Generation & Citations (Person 4 / Groq LLM)
* **Final Synthesis:**
  * ContextElements are structured dynamically into a system message format.
  * Uses Groq LLM (LLaMA-3.3-70B-Versatile) to synthesize grounded, JSON-parsed responses.
  * **Lineage-backed citations:** Generates precise citations matching element IDs, document sources, page numbers, and exact snippets.
* **Multi-Turn Memory:** Retains conversation history to handle contextual follow-up questions seamlessly.

---

## Slide 9: Validation Metrics & Results
* **Quarterly Report PDF Benchmark (18 nodes, 146 edges):**
  * **Retrieval Confidence:** `0.5832` (Semantic similarity + layout density).
  * **Multi-Modal Coherence Test:** Querying financial growth successfully retrieved the raw data table AND dynamically expanded to include the corresponding table caption block.
* **Performance:** Zero external API latency for retrieval; runs entirely local and in-memory using lightweight dependencies.

---

## Slide 10: Conclusion & Demo (Top 5 Call to Action)
* **Key Achievements:**
  * Dynamic, layout-aware document modeling.
  * Fully resolved schema integration layers.
  * Production-ready explainable citation chains.
  * Low infrastructure cost (runs locally, in-memory DB support).
* **Live Demo Showcase:** Show the console execution from PDF ingestion to grounded answer generation.
