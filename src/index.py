import json
import math
from pprint import pprint
import sys
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path

from .models import MinimalSource, Tokenizer


class Indexer:

    @classmethod
    def index(
        cls, max_chunk_size: int = 2000, overlap_size: int = 250
    ) -> None:
        # chunks = cls.__get_chunks(max_chunk_size)
        chunks = cls.__chunk(max_chunk_size, overlap_size)
        # pprint(chunks)
        # pprint(chunks)
        # index_objs: dict[str, dict] = {}

        # # store TF in (chunk_id, TF)
        # for chunk in chunks:
        #     for token in Tokenizer.tokenize(chunk["text"]):
        #         index_objs.setdefault(token, {"TF": []})["TF"].append(
        #             f'{chunk["id"]},{chunk["text"].count(token)}'
        #         )
        # chunks_len = len(chunks)

        # for token, value in index_objs.items():
        #     value["IDF"] = math.log(chunks_len / len(value["TF"]))

        # with open("data/processed/index.json", "w") as f:
        #     json.dump(index_objs, f)

        with open("data/processed/chunks.json", "w") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4)

    @classmethod
    def __chunk(cls, max_chunk_size: int, overlap_size: int):
        base_path = Path("data/raw")
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
        id = 0
        for path in base_path.rglob("*.md"):
            with open(path) as f:
                content = f.read()
                content_len = len(content)
                start_idx = 0
                end_idx = 0
                while start_idx <= content_len:

                    end_idx = min(start_idx + max_chunk_size, content_len) - 1
                    chunks_objs.append(
                        MinimalSource(
                            id=id,
                            file_path=str(path),
                            first_character_index=start_idx,
                            last_character_index=end_idx,
                        )
                    )
                    start_idx += max_chunk_size - overlap_size
                    id += 1

        return chunks_objs
