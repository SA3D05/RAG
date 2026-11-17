from enum import Enum


class Statics(Enum):
    MIN_CHUNK_SIZE = 500
    OVERLAP_SIZE = 300
    GLOBAL_MAX_CHUNK_SIZE = 2000


class PathInfo(Enum):

    CHUNKS_PATH = "data/processed/chunks.json"
    TF_PATH = "data/processed/term_frequency.json"
    IDF_PATH = "data/processed/inverse_document_frequency.json"
    AVGDL_PATH = "data/processed/average_document_length.json"
