"""
ingest.py — PDF ingestion phase using Docling.

FLOW:
    PDF → DocumentConverter → DoclingDocument → HybridChunker → inject_metadata → (text, metadata) tuples
"""

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from config import DocumentMetadata, BGE_M3_MODEL_PATH


def initialize_converter() -> DocumentConverter:
    return DocumentConverter()


def convert_pdf(converter: DocumentConverter, pdf_path: Path):
    result = converter.convert(str(pdf_path))
    return result.document


def chunk_document(docling_document) -> list:
    chunker = HybridChunker(
        tokenizer=str(BGE_M3_MODEL_PATH),
        max_tokens=512,
        merge_peers=True,
    )
    return list(chunker.chunk(docling_document))


def inject_metadata(chunks: list, document_metadata: DocumentMetadata) -> list[tuple[str, dict]]:
    result = []
    base_meta = document_metadata.model_dump()
    for chunk in chunks:
        merged = {**base_meta}
        try:
            merged["page_number"] = chunk.meta.doc_items[0].prov[0].page_no
        except (AttributeError, IndexError):
            merged["page_number"] = -1
        try:
            headings = chunk.meta.headings
            merged["section_heading"] = headings[0] if headings else ""
        except (AttributeError, IndexError):
            merged["section_heading"] = ""
        if chunk.text.strip():
            result.append((chunk.text, merged))
    return result


def ingest_document(pdf_path: Path, document_metadata: DocumentMetadata) -> list[tuple[str, dict]]:
    converter = initialize_converter()
    doc = convert_pdf(converter, pdf_path)
    chunks = chunk_document(doc)
    return inject_metadata(chunks, document_metadata)
