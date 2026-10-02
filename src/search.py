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
        # pprint(indexed_objs)

        for token in query_tokens:
            if idfs.get(token) is None:
                continue

            tf = tfs[token]
            idf = idfs[token]

            for chunk_id, value in tf.items():
                scores[chunk_id] = scores.get(chunk_id, 0) + (idf * value)

        top_relevant_ids: list = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]
        for id in top_relevant_ids:

            chunk = chunks[int(id)]
            result.append(chunk)

        return result
