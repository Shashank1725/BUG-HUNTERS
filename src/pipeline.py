"""
pipeline.py
-----------
Main entry point. Auto-detects file type (PDF / DOCX) and routes
to the correct parser, producing one unified ParsedDocument.

Usage:
  python pipeline.py document.pdf
  python pipeline.py document.docx
  python pipeline.py document.pdf --camelot     # higher-accuracy tables
  python pipeline.py folder/ --batch            # parse every file in a folder
"""

import os
import sys
import argparse
import time
from typing import List
from schema import DocElement, ElementType, ParsedDocument
from pdf_parser import parse_pdf
from docx_parser import parse_docx
from xlsx_parser import parse_xlsx
from csv_parser import parse_csv
from image_parser import parse_image


SUPPORTED_EXTENSIONS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".csv": parse_csv,
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image
}

_embed_model = None

def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        print("[*] Loading embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def generate_embeddings(elements: List[DocElement]):
    """Generate vector embeddings for every text/table/caption element."""
    model = get_embedding_model()
    texts = [e.content for e in elements if e.content]
    if not texts:
        return
    
    # Filter out extremely long tables or binary strings if any
    clean_texts = [str(t)[:1000] for t in texts] # clip for standard encoder
    
    print(f"[*] Generating embeddings for {len(clean_texts)} elements...")
    vectors = model.encode(clean_texts).tolist()
    
    # Map back to elements
    text_ptr = 0
    for e in elements:
        if e.content:
            e.embedding = vectors[text_ptr]
            text_ptr += 1


def parse_document(file_path: str, output_dir: str = "./output", use_camelot: bool = False) -> ParsedDocument:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}")

    images_dir = os.path.join(output_dir, "images")
    t0 = time.time()

    if ext == ".pdf":
        result = parse_pdf(file_path, output_dir=images_dir, use_camelot=use_camelot)
    elif ext == ".xlsx":
        result = parse_xlsx(file_path, output_dir=images_dir)
    elif ext == ".csv":
        result = parse_csv(file_path, output_dir=images_dir)
    elif ext in [".png", ".jpg", ".jpeg"]:
        result = parse_image(file_path, output_dir=images_dir)
    else:
        result = parse_docx(file_path, output_dir=images_dir)

    elapsed = time.time() - t0
    result.parse_stats["parse_time_seconds"] = round(elapsed, 2)
    
    # --- RAG ENHANCEMENT ---
    generate_embeddings(result.elements)
    # -----------------------
    
    return result


def batch_parse(folder_path: str, output_dir: str = "./output") -> list:
    results = []
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    print(f"[*] Batch parsing {len(files)} files from {folder_path}\n")

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            result = parse_document(fpath, output_dir=output_dir)
            doc_id = os.path.splitext(fname)[0]
            out_json = os.path.join(output_dir, f"{doc_id}_parsed.json")
            result.save(out_json)
            results.append(result)
            print(f"[✓] {fname}: {result.parse_stats['total_elements']} elements -> {out_json}\n")
        except Exception as e:
            print(f"[✗] {fname}: FAILED — {e}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Document Ingestion & Multi-Modal Parsing Pipeline")
    parser.add_argument("path", help="Path to a file (PDF/DOCX/XLSX/CSV/Image), or a folder if --batch")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--camelot", action="store_true", help="Use Camelot for higher-accuracy table extraction")
    parser.add_argument("--batch", action="store_true", help="Parse all supported files in a folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.batch:
        results = batch_parse(args.path, args.output)
        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE: {len(results)} documents parsed")
        print(f"{'='*60}")
        total_elements = sum(r.parse_stats["total_elements"] for r in results)
        print(f"  Total elements extracted: {total_elements}")
        return

    result = parse_document(args.path, args.output, use_camelot=args.camelot)

    out_name = os.path.splitext(os.path.basename(args.path))[0]
    out_path = os.path.join(args.output, f"{out_name}_parsed.json")
    result.save(out_path)

    print(f"\n{'='*60}")
    print(f"  PARSE COMPLETE")
    print(f"{'='*60}")
    print(f"  Source        : {args.path}")
    print(f"  Pages         : {result.total_pages}")
    print(f"  Total elements: {result.parse_stats['total_elements']}")
    print(f"  Breakdown     : {result.parse_stats['by_type']}")
    print(f"  Parse time    : {result.parse_stats['parse_time_seconds']}s")
    print(f"  Saved to      : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
