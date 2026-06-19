# Document Ingestion & Multi-Modal Parsing

**Person 1's module** — converts raw **PDF, DOCX, XLSX, CSV, and Image** files into a clean, structured, machine-readable representation. It extracts headings, paragraphs, tables, images, and captions — each tagged with page number, position, and type.

This is the **foundation layer** for any downstream RAG / LLM-context pipeline: garbage in, garbage out.

---

## What it does

```
Raw Files (PDF/DOCX/XLSX/CSV/PNG/JPG)
     │
     ▼
┌─────────────────────────────────────────┐
│  1. Layout-aware text extraction         │  PDF/DOCX: headings, lists
│     → headings, paragraphs, lists        │  heuristics for level detection
├─────────────────────────────────────────┤
│  2. Tabular Data extraction              │  Excel/CSV/PDF: 
│     → clean markdown tables              │  Pandas & pdfplumber & Camelot
├─────────────────────────────────────────┤
│  3. Image extraction + captioning        │  PDF/DOCX/Images extracts bytes
│     → auto-generated AI captions         │  BLIP generates captions
├─────────────────────────────────────────┤
│  4. Caption linking                      │  "Figure 1: ..." auto-linked
│     → captions matched to nearest figure │  to nearest image/table
└─────────────────────────────────────────┘
     │
     ▼
Unified JSON output (ParsedDocument schema)
```

---

## Supported Formats

| Format | Parsing Method | Features |
|---|---|---|
| **PDF** | PyMuPDF + pdfplumber | Layout detection, font-size heuristics, table extraction |
| **DOCX** | python-docx | Style-aware parsing, image extraction |
| **XLSX** | pandas + openpyxl | Multi-sheet support, Markdown table conversion |
| **CSV** | pandas | Data cleaning, Markdown table conversion |
| **Images**| BLIP (AI Model) | PNG/JPG/JPEG support with auto-AI captioning |

---

## Output Schema

Every element — regardless of source — is normalized to a unified schema:

```json
{
  "element_id": "report_p1_el3",
  "type": "table",
  "content": "| Metric | Value |\n| --- | --- |\n| Revenue | $10M |",
  "page": 1,
  "heading_level": null,
  "metadata": {"rows": 5, "cols": 2, "extraction_method": "pandas"}
}
```

---

## Quick Start

### Installation
```bash
# Recommended for Python 3.13
py -3.13 -m pip install -r requirements.txt
```

### Usage
```bash
# Parse a PDF or DOCX
py -3.13 src/pipeline.py document.pdf

# Parse an Excel or CSV file
py -3.13 src/pipeline.py data.xlsx
py -3.13 src/pipeline.py results.csv

# Parse a standalone Image
py -3.13 src/pipeline.py photo.jpg

# Batch parse a whole folder
py -3.13 src/pipeline.py folder_path/ --batch
```

---

## File Structure

```
doc_ingestion/
├── src/
│   ├── schema.py          # Unified DocElement dataclasses
│   ├── pdf_parser.py       # PDF parsing logic
│   ├── docx_parser.py      # Word parsing logic
│   ├── xlsx_parser.py      # Excel parsing logic
│   ├── csv_parser.py       # CSV parsing logic
│   ├── image_parser.py     # Standalone image parsing
│   ├── captioner.py        # BLIP image captioning AI
│   └── pipeline.py         # Main orchestrator
├── requirements.txt
└── README.md
```

---

## AI Features (Image Captioning)
The pipeline uses **Salesforce/blip-image-captioning-base** (~990MB). The first time you run an image parse, the model will download automatically. It runs locally on CPU/GPU without recurring costs.
