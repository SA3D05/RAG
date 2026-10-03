import json
from pydantic import TypeAdapter
from .models import MinimalSource, Tokenizer


class Retrieval:

    @classmethod
    def search(cls, query: str, k: int) -> list[MinimalSource]:
        query_tokens = Tokenizer.tokenize(query)
        # print(query_tokens)
        result: list[MinimalSource] = []
        scores = {}
        idfs: dict = {}
        tfs: dict = {}

        chunks: list[MinimalSource] = []
        with open("data/processed/chunks.json") as f:
            chunks = TypeAdapter(list[MinimalSource]).validate_json(f.read())

        with open("data/processed/tf.json") as f:
            tfs = json.load(f)

        with open("data/processed/idf.json") as f:
            idfs = json.load(f)

        avgdl = 0
        with open("data/processed/avgdl.json") as f:
            avgdl = json.load(f)["avgdl"]
        # pprint(indexed_objs)

        for token in query_tokens:
            if idfs.get(token) is None:
                continue

            tf = tfs[token]
            idf = idfs[token]

            for chunk_id, tf_value in tf.items():
                scores[chunk_id] = scores.get(chunk_id, 0) + cls.__score_chunk(
                    tf_value,
                    idf,
                    chunks[int(chunk_id)].tokens_len,
                    avgdl,
                    "bm25",
                )

        top_relevant_ids: list = sorted(
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
        method: str = "bm25",
    ) -> float:
        if method == "tfidf":
            return tf * idf

        # Default to BM25
        k1 = 1.5
        b = 0.75
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
        return idf * (numerator / denominator)
