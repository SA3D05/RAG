from enum import Enum, auto
from pathlib import Path


class DocumentType(Enum):
    CODE = auto()
    DOCS = auto()


class Statics(Enum):
    CODE_MIN_CHUNK_SIZE = 500
    DOCS_OVERLAP_SIZE = 300


class PathInfo(Enum):

    CODE_TF = "data/processed/code_term_frequency.json"
    CODE_IDF = "data/processed/code_inverse_document_frequency.json"

    CODE_AVGDL = "data/processed/code_average_document_length.json"
    CODE_CHUNKS = "data/processed/code_chunks.json"

    DOCS_TF = "data/processed/docs_term_frequency.json"
    DOCS_IDF = "data/processed/docs_inverse_document_frequency.json"

    DOCS_AVGDL = "data/processed/docs_average_document_length.json"
    DOCS_CHUNKS = "data/processed/docs_chunks.json"
