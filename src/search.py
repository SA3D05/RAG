from pathlib import Path
from .utils import Utils
from .enums import PathInfo
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

    def search(self, query: str, k: int) -> list[MinimalSource]:

        query_tokens = Utils.tokenize(query)
        scores = {}
        result: list[MinimalSource] = []

        for token in query_tokens:

            if (
                token not in self.inverse_document_frequency.idf
            ):  # skip unknown tokens in query
                continue

            token_frequency = self.term_frequency.tf[token]
            inverse_document_frequency = self.inverse_document_frequency.idf[
                token
            ]

            for chunk_id, tf_value in token_frequency.items():

                doc_len = self.chunks.chunks[int(chunk_id)].tokens_len

                chunk_score = self.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    doc_len,
                    self.avrage_document_length.avgdl,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = self.chunks.chunks[int(id)]
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
