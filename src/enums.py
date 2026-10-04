from enum import Enum, auto
from pathlib import Path


class DocumentType(Enum):
    CODE = auto()
    DOCS = auto()


class PathInfo(Enum):

    CODE_TF = Path("data/processed/code_term_frequency.json")
    CODE_IDF = Path("data/processed/code_inverse_document_frequency.json")

    CODE_AVGDL = Path("data/processed/code_average_document_length.json")
    CODE_CHUNKS = Path("data/processed/code_chunks.json")

    DOCS_TF = Path("data/processed/docs_term_frequency.json")
    DOCS_IDF = Path("data/processed/docs_inverse_document_frequency.json")

    DOCS_AVGDL = Path("data/processed/docs_average_document_length.json")
    DOCS_CHUNKS = Path("data/processed/docs_chunks.json")

    RAW_DATA = Path("data/raw/vllm-0.10.1")
