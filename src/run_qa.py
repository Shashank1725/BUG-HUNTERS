"""
run_qa.py — End-to-End Runner
Usage:
    python run_qa.py <path_to_pdf_or_any_file>

What it does:
    1. Runs Person 1's pipeline.py on the file → produces *_parsed.json
    2. Loads all elements from the parsed JSON
    3. Starts an interactive QA session powered by Groq LLM
"""

import os
import sys
import json
import subprocess
import glob

from qa_engine import QAEngine, ContextElement, result_to_dict, _confidence_label


def run_pipeline(file_path: str) -> str:
    """
    Call Person 1's pipeline.py on the given file.
    Returns the path to the output *_parsed.json file.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(src_dir, "pipeline.py")
    output_dir = os.path.join(src_dir, "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nParsing document: {os.path.basename(file_path)}")
    print("This may take a moment...\n")

    result = subprocess.run(
        [sys.executable, pipeline_path, file_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Pipeline error:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Find the output JSON that matches this file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    matches = glob.glob(os.path.join(output_dir, f"{base_name}*_parsed.json"))

    if not matches:
        # Fallback: get the most recently created parsed JSON
        all_jsons = glob.glob(os.path.join(output_dir, "*_parsed.json"))
        if not all_jsons:
            print("No parsed JSON found in output/ folder.")
            sys.exit(1)
        matches = [max(all_jsons, key=os.path.getctime)]

    return matches[0]


def load_context(json_path: str) -> list[ContextElement]:
    """Load all elements from Person 1's parsed JSON output."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = os.path.basename(data.get("source_file", json_path))
    elements = []

    for el in data.get("elements", []):
        # content = el.get("content", "").strip()
        raw = el.get("content", "")
        # content = " ".join(raw) if isinstance(raw, list) else str(raw).strip()
        # NEW
        content = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw).strip()
        content = content.strip()
        if not content:
            continue
        elements.append(ContextElement(
            element_id=el["element_id"],
            type=el.get("type", "paragraph"),
            content=content,
            page=el.get("page", 1),
            doc_id=doc_id,
            # relevance_score=el.get("confidence", 1.0),
            # NEW
            relevance_score=float(el.get("confidence") or 1.0),
        ))

    return elements


def run_interactive_qa(elements: list[ContextElement], doc_name: str):
    """Interactive QA loop — type questions, get answers."""
    engine = QAEngine()

    print("=" * 60)
    print(f"Document: {doc_name}")
    print(f"Elements loaded: {len(elements)}")
    print("Type your question (or 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        print("\nThinking...")
        result = engine.ask(question, elements)

        print(f"\nAnswer: {result.answer if result.is_answerable else '[Cannot answer — insufficient context]'}")
        print(f"Confidence: {_confidence_label(result.confidence)} ({result.confidence:.0%})")

        if result.citations:
            print("\nCitations:")
            for c in result.citations:
                print(f"  ✓ [{c.type.upper()}] Page {c.page} — \"{c.snippet}\"")

        if result.reasoning_path:
            print(f"\nReasoning path: {' → '.join(result.reasoning_path)}")

        if result.missing_evidence:
            print(f"\nMissing: {result.missing_evidence}")

        print("-" * 60)


if __name__ == "__main__":
    # Check API key
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set.")
        print("Run: $env:GROQ_API_KEY='gsk_your-key-here'")
        sys.exit(1)

    # Get file path from argument or ask user
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to PDF/DOCX/Image file: ").strip().strip('"')

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Step 1: Parse the document
    parsed_json_path = run_pipeline(file_path)
    print(f"Parsed output: {parsed_json_path}")

    # Step 2: Load elements
    elements = load_context(parsed_json_path)
    if not elements:
        print("No content elements found in parsed output.")
        sys.exit(1)

    # Step 3: Start QA
    run_interactive_qa(elements, os.path.basename(file_path))