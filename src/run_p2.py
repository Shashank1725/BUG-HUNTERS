# # run_p2.py  ← create this file in your project folder and run it

# from pipeline import parse_document        # Person 1's pipeline
# from p2_pipeline import GraphPipeline, PipelineConfig

# # Step 1: Person 1 parses the document
# parsed_doc = parse_document("C:\Users\peric\ORACLE\SET2M5.pdf", output_dir="./output")

# # Step 2: Person 2 builds the graph
# config = PipelineConfig(
#     output_dir="./output",          # saves doc_graph.pkl + doc_graph.json
#     use_ml_similarity=False,        # set True if sentence-transformers installed
#     use_entity_cooccurrence=False,  # set True if spaCy installed
# )
# graph = GraphPipeline(config).run_from_doc(parsed_doc)

# # Step 3: Inspect the result
# print(graph.stats())

# # Step 4: Try the query helpers (what Person 3 will call)
# all_nodes = graph.all_nodes()
# if all_nodes:
#     seed_id = all_nodes[0].element_id
#     context = graph.subgraph_around(seed_id, hops=2).all_nodes()
#     print(f"\nContext bundle around '{seed_id}': {len(context)} elements")
#     for node in context:
#         print(f"  [{node.type.value}] {node.content[:80]}")

# run_p2.py

# import sys
# from pipeline import parse_document
# from p2_pipeline import GraphPipeline, PipelineConfig

# if len(sys.argv) < 2:
#     print("Usage: python run_p2.py <path_to_pdf>")
#     print("Example: python run_p2.py report.pdf")
#     sys.exit(1)

# pdf_path = sys.argv[1]

# # Step 1: Person 1 parses the document
# parsed_doc = parse_document(pdf_path, output_dir="./output")

# # Step 2: Person 2 builds the graph
# config = PipelineConfig(
#     # output_dir="./output",
#     # run_p2.py  — change this one line
# output_dir  = sys.argv[2] if len(sys.argv) > 2 else "../output",
#     use_ml_similarity=False,
#     use_entity_cooccurrence=False,
# )
# graph = GraphPipeline(config).run_from_doc(parsed_doc)

# # Step 3: Print stats
# print(graph.stats())

# # Step 4: Show context bundle for first element
# all_nodes = graph.all_nodes()
# if all_nodes:
#     seed_id = all_nodes[0].element_id
#     context = graph.subgraph_around(seed_id, hops=2).all_nodes()
#     print(f"\nContext bundle around '{seed_id}': {len(context)} elements")
#     for node in context:
#         print(f"  [{node.type.value}] {node.content[:80]}")

# run_p2.py

# import sys
# import os
# from pipeline import parse_document
# from p2_pipeline import GraphPipeline, PipelineConfig

# if len(sys.argv) < 2:
#     print("Usage: python run_p2.py <path_to_pdf> [output_folder]")
#     sys.exit(1)

# pdf_path   = sys.argv[1]
# output_dir = sys.argv[2] if len(sys.argv) > 2 else "../output"

# # Make sure output folder exists
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

# # Step 1: Person 1 parses — images go to output_dir/images/
# parsed_doc = parse_document(pdf_path, output_dir=output_dir)

# # Step 2: Person 2 builds graph — saves doc_graph.pkl + doc_graph.json
# config = PipelineConfig(
#     output_dir=output_dir,
#     use_ml_similarity=False,
#     use_entity_cooccurrence=False,
# )
# graph = GraphPipeline(config).run_from_doc(parsed_doc)

# # Step 3: Print stats
# print("\n" + "="*60)
# print("  GRAPH STATS")
# print("="*60)
# stats = graph.stats()
# print(f"  Total nodes : {stats['total_nodes']}")
# print(f"  Total edges : {stats['total_edges']}")
# print(f"  Node types  : {stats['node_types']}")
# print(f"  Edge types  : {stats['edge_types']}")
# print(f"\n  Output saved to: {os.path.abspath(output_dir)}")
# print("="*60)

# """
# run_p2.py  ─  Person 2 runner
# ------------------------------
# Usage:
#   python run_p2.py <pdf_path> [output_folder]


# """

# import sys
# import os
# from pipeline       import parse_document
# from p2_pipeline    import GraphPipeline, PipelineConfig
# from visualize_graph import visualize

# if len(sys.argv) < 2:
#     print("Usage: python run_p2.py <path_to_pdf> [output_folder]")
#     sys.exit(1)

# pdf_path   = sys.argv[1]
# output_dir = sys.argv[2] if len(sys.argv) > 2 else "../output"

# # Create output folders
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

# # ── Step 1: Person 1 — Parse the PDF ─────────────────────────────────────────
# print("\n" + "="*60)
# print("  STEP 1: Parsing PDF")
# print("="*60)
# parsed_doc = parse_document(pdf_path, output_dir=output_dir)

# # ── Step 2: Person 2 — Build the graph ───────────────────────────────────────
# print("\n" + "="*60)
# print("  STEP 2: Building Relationship Graph")
# print("="*60)
# config = PipelineConfig(
#     output_dir=output_dir,
#     use_ml_similarity=False,
#     use_entity_cooccurrence=False,
# )
# graph = GraphPipeline(config).run_from_doc(parsed_doc)

# # ── Step 3: Person 2 — Visualize the graph ───────────────────────────────────
# print("\n" + "="*60)
# print("  STEP 3: Generating Graph Visualizations")
# print("="*60)
# visualize(os.path.join(output_dir, "doc_graph.pkl"), output_dir)

# # ── Summary ───────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("  COMPLETE — Output Files")
# print("="*60)
# stats = graph.stats()
# print(f"  PDF parsed     : {os.path.basename(pdf_path)}")
# print(f"  Total nodes    : {stats['total_nodes']}")
# print(f"  Total edges    : {stats['total_edges']}")
# print(f"  Node types     : {stats['node_types']}")
# print(f"  Edge types     : {stats['edge_types']}")
# print(f"\n  Output folder  : {os.path.abspath(output_dir)}")
# print(f"  ├── images/              ← extracted images from PDF")
# print(f"  ├── doc_graph.pkl        ← graph for Person 3 (retrieval)")
# print(f"  ├── doc_graph.json       ← human-readable graph")
# print(f"  ├── graph_full.png       ← full graph diagram")
# print(f"  ├── graph_by_type.png    ← nodes & edges by type")
# print(f"  ├── graph_per_page.png   ← per page view")
# print(f"  └── graph_stats.png      ← statistics bar chart")
# print("="*60)

import sys
import os
import shutil
from pipeline        import parse_document
from p2_pipeline     import GraphPipeline, PipelineConfig
from visualize_graph import visualize

if len(sys.argv) < 2:
    print("Usage: python run_p2.py <path_to_pdf> [output_folder]")
    sys.exit(1)

pdf_path   = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else "../output"
images_dir = os.path.join(output_dir, "images")   # output/images/

os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# ── Step 1: Parse PDF ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: Parsing PDF")
print("="*60)
parsed_doc = parse_document(pdf_path, output_dir=output_dir)

# Move any stray images from output/ into output/images/
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
moved = 0
for fname in os.listdir(output_dir):
    ext = os.path.splitext(fname)[1].lower()
    if ext in IMAGE_EXTS:
        src  = os.path.join(output_dir, fname)
        dest = os.path.join(images_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, dest)
            moved += 1
if moved:
    print(f"[*] Moved {moved} image(s) → images/")

# ── Step 2: Build Graph ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: Building Relationship Graph")
print("="*60)
config = PipelineConfig(
    output_dir=output_dir,
    use_ml_similarity=False,
    use_entity_cooccurrence=False,
)
graph = GraphPipeline(config).run_from_doc(parsed_doc)

# ── Step 3: Visualize → save into output/images/ ─────────────────────────────
print("\n" + "="*60)
print("  STEP 3: Generating Graph Visualizations")
print("="*60)
visualize(os.path.join(output_dir, "doc_graph.pkl"), images_dir)  # ← images_dir

# ── Summary ───────────────────────────────────────────────────────────────────
stats   = graph.stats()
abs_out = os.path.abspath(output_dir)
print("\n" + "="*60)
print("  COMPLETE — Output Files")
print("="*60)
print(f"  PDF parsed    : {os.path.basename(pdf_path)}")
print(f"  Total nodes   : {stats['total_nodes']}")
print(f"  Total edges   : {stats['total_edges']}")
print(f"  Node types    : {stats['node_types']}")
print(f"  Edge types    : {stats['edge_types']}")
print(f"\n  Output folder : {abs_out}")
print(f"  ├── images/")
print(f"  │   ├── SET2M5_img0.png     ← extracted PDF images")
print(f"  │   ├── graph_full.png      ← full graph diagram")
print(f"  │   ├── graph_by_type.png   ← nodes & edges by type")
print(f"  │   ├── graph_per_page.png  ← per page view")
print(f"  │   └── graph_stats.png     ← statistics bar chart")
print(f"  ├── doc_graph.pkl           ← graph for Person 3")
print(f"  └── doc_graph.json          ← human-readable graph")
print("="*60)