import json
from pprint import pprint

from pydantic import TypeAdapter

from .models import MinimalSource, Tokenizer


class Retrieval:

    @classmethod
    def search(cls, query: str, k: int) -> list[MinimalSource]:
        query_tokens = Tokenizer.tokenize(query)
        print(query_tokens)
        result: list[MinimalSource] = []
        scores = {}
        indexed_objs: dict[str, dict[str, list | float]] = {}

        chunks: list[MinimalSource] = []
        with open("data/processed/chunks.json") as f:
            chunks = TypeAdapter(list[MinimalSource]).validate_json(f.read())

        with open("data/processed/index.json") as f:
            indexed_objs = json.load(f)
        # pprint(indexed_objs)
        for token in query_tokens:
            if indexed_objs.get(token) is None:
                continue

            tfs = {}
            idf: float = 0
            for tfidf, value in indexed_objs[token].items():
                if tfidf == "TF":
                    for chunk_tf in value:  # type: ignore
                        chunk_id, tf = chunk_tf.split(",")
                        tfs[chunk_id] = int(tf)
                else:
                    idf = value  # type: ignore

            for chunk_id, tf in tfs.items():
                scores[chunk_id] = scores.get(chunk_id, 0) + tf * idf
        top_relevant_ids: list = sorted(
            scores.keys(), key=lambda id: scores[id]
        )[:k]
        for id in top_relevant_ids:

            chunk = chunks[int(id)]
            result.append(chunk)

        return result
