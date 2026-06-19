"""
csv_parser.py
-------------
CSV parsing using pandas.
Extracts CSV data into a structured TABLE element in Markdown format.
"""

import os
import pandas as pd
from tabulate import tabulate
from typing import List
from schema import DocElement, ElementType, ParsedDocument


def parse_csv(file_path: str, output_dir: str = "./output/images") -> ParsedDocument:
    """
    Parses a CSV file and converts it into a Markdown table.
    """
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"[*] Parsing CSV: {file_path}")
    
    elements: List[DocElement] = []
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        if not df.empty:
            # Convert DataFrame to Markdown table
            md_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            elements.append(DocElement(
                element_id=f"{doc_id}_csv_table",
                type=ElementType.TABLE,
                content=md_table,
                page=1,
                metadata={
                    "rows": len(df),
                    "cols": len(df.columns),
                    "extraction_method": "pandas+tabulate"
                }
            ))
    except Exception as e:
        print(f"[!] Error opening CSV {file_path}: {e}")
        raise ValueError(f"Could not read CSV file: {e}")

    parsed = ParsedDocument(
        source_file=file_path,
        doc_type="csv",
        total_pages=1,
        elements=elements,
        parse_stats={
            "total_elements": len(elements),
            "by_type": {},
        },
    )
    parsed.parse_stats["by_type"] = parsed.summary()
    return parsed

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python csv_parser.py <path_to_csv>")
        sys.exit(1)
    
    result = parse_csv(sys.argv[1])
    print(f"\n[✓] Parsed {len(result.elements)} table from {result.source_file}")
