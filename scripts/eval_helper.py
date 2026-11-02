"""Preview extracted PDF pages to help write grounded evaluation questions."""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    """Print each PDF's page number and first 200 extracted characters."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}")
        return
    for pdf_path in pdfs:
        print(f"\n## {pdf_path.name}")
        try:
            with pymupdf.open(pdf_path) as document:
                for page_number, page in enumerate(document, start=1):
                    preview = " ".join(page.get_text("text").split())[:200]
                    print(f"Page {page_number}: {preview}")
        except Exception as error:
            print(f"Could not read PDF: {type(error).__name__}")


if __name__ == "__main__":
    main()
