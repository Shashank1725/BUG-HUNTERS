"""
xlsx_parser.py
--------------
Excel parsing using pandas and openpyxl.
Extracts each sheet as a structured TABLE element in Markdown format.
"""

import os
import pandas as pd
from tabulate import tabulate
from typing import List
from schema import DocElement, ElementType, ParsedDocument


def parse_xlsx(file_path: str, output_dir: str = "./output/images") -> ParsedDocument:
    """
    Parses an Excel file, converting each sheet into a Markdown table.
    """
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"[*] Parsing XLSX: {file_path}")
    
    elements: List[DocElement] = []
    elem_counter = 0
    
    try:
        # Load the Excel file
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
    except Exception as e:
        print(f"[!] Error opening XLSX {file_path}: {e}")
        raise ValueError(f"Could not read Excel file: {e}")

    for sheet_name in sheet_names:
        try:
            # Read sheet into DataFrame
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Skip empty sheets
            if df.empty:
                continue
                
            elem_counter += 1
            
            # Convert DataFrame to Markdown table
            # 'pipe' format is standard Github-Flavored Markdown
            md_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            elements.append(DocElement(
                element_id=f"{doc_id}_s{elem_counter}_table",
                type=ElementType.TABLE,
                content=md_table,
                page=1,  # Sheets aren't strictly pages, but we default to 1
                metadata={
                    "sheet_name": sheet_name,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "extraction_method": "pandas+tabulate"
                }
            ))
        except Exception as e:
            print(f"[!] Warning: Failed to parse sheet '{sheet_name}' in {file_path}: {e}")

    parsed = ParsedDocument(
        source_file=file_path,
        doc_type="xlsx",
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
    import json
    if len(sys.argv) < 2:
        print("Usage: python xlsx_parser.py <path_to_xlsx>")
        sys.exit(1)

    try:
        result = parse_xlsx(sys.argv[1])
        print(f"\n[✓] Parsed {len(result.elements)} tables from {result.source_file}")
        
        # Save a sample output if run directly
        os.makedirs("./output", exist_ok=True)
        out_path = "./output/sample_xlsx_parsed.json"
        result.save(out_path)
        print(f"[✓] Saved results to {out_path}")
    except Exception as e:
        print(f"[✗] Failed: {e}")
