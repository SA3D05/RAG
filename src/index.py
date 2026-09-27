import json
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from .models import MinimalSource, Tokenizer


class Indexer:

    @classmethod
    def index(cls, max_chunk_size: int = 2000) -> None:
        chunks = cls.__get_chunks(max_chunk_size)
        index_objs: dict[str, dict] = {}

        # store TF in (chunk_id, TF)
        for chunk in chunks:
            for token in Tokenizer.tokenize(chunk["text"]):
                index_objs.setdefault(token, {"TF": []})["TF"].append(
                    f'{chunk["id"]},{chunk["text"].count(token)}'
                )
        chunks_len = len(chunks)

        for token, value in index_objs.items():
            value["IDF"] = math.log(chunks_len / len(value["TF"]))

        with open("data/processed/index.json", "w") as f:
            json.dump(index_objs, f)

        with open("data/processed/chunks.json", "w") as f:
            json.dump(chunks, f)

    @classmethod
    def __get_chunks(cls, max_chunk_size: int) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
        )
        base_path = Path("data/raw")

        i = 0
        chunks_objs = []

        for path in base_path.rglob("*.md"):
            with open(path) as f:
                chunks = splitter.split_text(f.read().lower())
                for chunk in chunks:
                    chunks_objs.append(
                        MinimalSource(
                            id=i,
                            file_path=str(path),
                            text=chunk,
                            first_character_index=0,
                            last_character_index=0,
                        ).model_dump()
                    )
                    i += 1
        return chunks_objs
