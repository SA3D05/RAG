import json
import math
from pprint import pprint
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path

from .models import MinimalSource, Tokenizer


class Indexer:

    @classmethod
    def index(
        cls, max_chunk_size: int = 2000, overlap_size: int = 200
    ) -> None:
        # chunks = cls.__get_chunks(max_chunk_size)
        chunks = cls.__chunk(max_chunk_size, overlap_size)
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

    # @classmethod
    # def __get_chunks(cls, max_chunk_size: int) -> list[dict]:
    #     splitter = MarkdownHeaderTextSplitter(
    #         [("#", "H1"), ("##", "H2"), ("###", "H3")]
    #     )
    #     base_path = Path("data/raw")

    #     i = 0
    #     chunks_objs = []

    #     for path in base_path.rglob("*.md"):
    #         with open(path) as f:
    #             chunks = splitter.split_text(f.read().lower())
    #             for chunk in chunks:
    #                 chunks_objs.append(
    #                     MinimalSource(
    #                         id=i,
    #                         file_path=str(path),
    #                         text=f"{chunk.metadata} {chunk.page_content}",
    #                         first_character_index=0,
    #                         last_character_index=0,
    #                     ).model_dump()
    #                 )
    #                 i += 1
    #     return chunks_objs

    @classmethod
    def __chunk(cls, max_chunk_size: int, overlap_size: int):
        base_path = Path()
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
        id = 0

        def add_chunk():
            chunks_objs.append(
                MinimalSource(
                    id=id,
                    file_path=str(path),
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                )
            )

        for path in base_path.rglob("*.md"):
            start_idx = 0
            end_idx = 0
            with open(path) as f:
                content = f.read()
                content_len = len(content)

                while content_len > max_chunk_size:
                    if end_idx != 0:
                        end_idx += max_chunk_size - overlap_size
                        start_idx -= overlap_size
                    else:
                        end_idx += max_chunk_size
                        start_idx = end_idx
                    add_chunk()
                    id += 1
                    content_len -= max_chunk_size

                end_idx = start_idx + (content_len - 1)
                add_chunk()
                id += 1

        return chunks_objs
