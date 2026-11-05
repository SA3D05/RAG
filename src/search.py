import sys
from pydantic import ValidationError
from .enums import PathInfo, DocumentType
from .models import (
    MinimalSource,
    Tokenizer,
    FileOP,
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
        current_file = ""

        try:
            current_file = PathInfo.CHUNKS_PATH.value

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
                PathInfo,
            ):
                attr_name, attr_type = attr
                current_file = path
                setattr(
                    self,
                    attr_name,
                    FileOP.get_pydantic_model(path.value, attr_type),
                )

        except ValidationError as e:
            print("File:", current_file)
            for error in e.errors():
                print(error)
            sys.exit(1)

    def search(
        self, query: str, k: int, query_type: DocumentType
    ) -> list[MinimalSource]:

        if query_type == DocumentType.DOCS:
            query_tokens = Tokenizer.tokenize(query)
            result = self.__search_docs(query_tokens, k)
            return result

        query_tokens = Tokenizer.tokenize(query)
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

                if doc_len is None:
                    print("File:", PathInfo.CHUNKS_PATH.value)
                    print(
                        "Error: missing 'tokens_len' field in chunk index:",
                        chunk_id,
                    )
                    sys.exit(1)

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
                    print("File:", PathInfo.CHUNKS_PATH.value)
                    print(
                        "Error: missing 'tokens_len' field in chunk index:",
                        chunk_id,
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
        k1 = 1.5
        b = 0.75

        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))

        return idf * (numerator / denominator)
