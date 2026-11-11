from pathlib import Path
import sys
from .enums import PathInfo, DocumentType
from .utils import Utils
from .models import (
    MinimalSource,
    ChunksCache,
    TermFrequencyCache,
    InverseDocumentFrequencyCache,
    AvrageDocumentLengthCache,
)


class Retrieval:
    def __init__(self) -> None:
        self.chunks: ChunksCache
        self.term_frequency: TermFrequencyCache
        self.inverse_document_frequency: InverseDocumentFrequencyCache
        self.avrage_document_length: AvrageDocumentLengthCache
        self.__get_parsed_attrs()

    def __get_parsed_attrs(self):

        for attr, path in zip(
            [
                ("chunks", ChunksCache),
                ("term_frequency", TermFrequencyCache),
                (
                    "inverse_document_frequency",
                    InverseDocumentFrequencyCache,
                ),
                ("avrage_document_length", AvrageDocumentLengthCache),
            ],
            [
                PathInfo.CHUNKS_PATH.value,
                PathInfo.TF_PATH.value,
                PathInfo.IDF_PATH.value,
                PathInfo.AVGDL_PATH.value,
            ],
        ):
            attr_name, attr_type = attr
            setattr(
                self,
                attr_name,
                Utils.get_pydantic_model(Path(path), attr_type),
            )

    def search(
        self, query: str, k: int, query_type: DocumentType
    ) -> list[MinimalSource]:

        if query_type == DocumentType.DOCS:
            query_tokens = Utils.tokenize(query)
            result = self.__search_docs(query_tokens, k)
            return result

        query_tokens = Utils.tokenize(query)
        result = self.__search_code(query_tokens, k)
        return result

    def __search_docs(
        self, query_tokens: list[str], k: int
    ) -> list[MinimalSource]:
        scores = {}
        result: list[MinimalSource] = []

        for token in query_tokens:

            if (
                token not in self.inverse_document_frequency.docs_idf
            ):  # skip unknown tokens in query
                continue

            token_frequency = self.term_frequency.docs_tf[token]
            inverse_document_frequency = (
                self.inverse_document_frequency.docs_idf[token]
            )

            for chunk_id, tf_value in token_frequency.items():

                doc_len = self.chunks.docs_chunks[int(chunk_id)].tokens_len

                chunk_score = self.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    doc_len,
                    self.avrage_document_length.docs_avgdl,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = self.chunks.docs_chunks[int(id)]
            result.append(chunk)

        return result

    def __search_code(
        self, query_tokens: list[str], k: int
    ) -> list[MinimalSource]:
        scores = {}
        result: list[MinimalSource] = []

        for token in query_tokens:

            if (
                token not in self.inverse_document_frequency.code_idf
            ):  # skip unknown tokens in query
                continue

            token_frequency = self.term_frequency.code_tf[token]
            inverse_document_frequency = (
                self.inverse_document_frequency.code_idf[token]
            )

            for chunk_id, tf_value in token_frequency.items():

                doc_len = self.chunks.code_chunks[int(chunk_id)].tokens_len

                if doc_len is None:
                    Utils.write_err(f"File: {PathInfo.CHUNKS_PATH.value}")
                    Utils.write_err(
                        f"Error: missing 'tokens_len' field in chunk index: {chunk_id}",
                    )
                    sys.exit(1)

                chunk_score = self.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    doc_len,
                    self.avrage_document_length.code_avgdl,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = self.chunks.code_chunks[int(id)]
            result.append(chunk)

        return result

    def __score_chunk(
        self,
        tf: int,
        idf: float,
        doc_len: int,
        avgdl: float,
    ) -> float:
        if avgdl <= 0:
            return 0.0

        k1 = 1.5
        b = 0.75

        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))

        return idf * (numerator / denominator)
