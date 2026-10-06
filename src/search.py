import json
from pydantic import TypeAdapter
from .enums import PathInfo, DocumentType
from .models import MinimalSource, Tokenizer, tokenize_python_code


class Retrieval:

    @classmethod
    def search(
        cls, query: str, k: int, query_type: DocumentType
    ) -> list[MinimalSource]:

        if query_type == DocumentType.DOCS:
            query_tokens = Tokenizer.tokenize(query)
            result = cls.__search_docs(query_tokens, k)
            return result

        query_tokens = tokenize_python_code(query)
        result = cls.__search_code(query_tokens, k)
        return result

    @classmethod
    def __search_docs(
        cls, query_tokens: list[str], k: int
    ) -> list[MinimalSource]:
        scores = {}
        result: list[MinimalSource] = []
        AVGDL = 0
        IDF: dict[str, float] = {}
        TF: dict[str, dict[str, int]] = {}

        chunks: list[MinimalSource] = []

        with open(PathInfo.DOCS_CHUNKS.value) as f:
            chunks = TypeAdapter(list[MinimalSource]).validate_json(f.read())

        with open(PathInfo.DOCS_TF.value) as f:
            TF = json.load(f)

        with open(PathInfo.DOCS_IDF.value) as f:
            IDF = json.load(f)

        with open(PathInfo.DOCS_AVGDL.value) as f:
            AVGDL = json.load(f)["avgdl"]

        for token in query_tokens:

            if token not in IDF:  # skip unknown tokens in query
                continue

            token_frequency = TF[token]
            inverse_document_frequency = IDF[token]

            for chunk_id, tf_value in token_frequency.items():

                chunk_score = cls.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    chunks[int(chunk_id)].tokens_len,
                    AVGDL,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = chunks[int(id)]
            result.append(chunk)

        return result

    @classmethod
    def __search_code(
        cls, query_tokens: list[str], k: int
    ) -> list[MinimalSource]:
        scores = {}
        result: list[MinimalSource] = []
        AVGDL = 0
        IDF: dict[str, float] = {}
        TF: dict[str, dict[str, int]] = {}

        chunks: list[MinimalSource] = []

        with open(PathInfo.CODE_CHUNKS.value) as f:
            chunks = TypeAdapter(list[MinimalSource]).validate_json(f.read())

        with open(PathInfo.CODE_TF.value) as f:
            TF = json.load(f)

        with open(PathInfo.CODE_IDF.value) as f:
            IDF = json.load(f)

        with open(PathInfo.CODE_AVGDL.value) as f:
            AVGDL = json.load(f)["avgdl"]

        for token in query_tokens:

            if token not in IDF:  # skip unknown tokens in query
                continue

            token_frequency = TF[token]
            inverse_document_frequency = IDF[token]

            for chunk_id, tf_value in token_frequency.items():

                chunk_score = cls.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    chunks[int(chunk_id)].tokens_len,
                    AVGDL,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = chunks[int(id)]
            result.append(chunk)

        return result

    @classmethod
    def __score_chunk(
        cls,
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
