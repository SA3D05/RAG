import json
import math

from tqdm import tqdm
from .enums import DocumentType, PathInfo

from .models import MinimalSource, Tokenizer
from collections import Counter


class Indexer:

    @classmethod
    def index(cls, max_chunk_size: int) -> None:

        print("chunking ...")
        docs_chunks = cls.__chunk_docs(max_chunk_size, 300)
        code_chunks = cls.__chunk_code(max_chunk_size)

        print("indexing ...")
        with tqdm(total=len(code_chunks) + len(docs_chunks)) as pbar:
            docs_index = cls.__get_chunks_index(docs_chunks, pbar)
            code_index = cls.__get_chunks_index(code_chunks, pbar)

        print("saving ...")
        with tqdm(total=8) as pbar:
            cls.__save_index(docs_chunks, docs_index, DocumentType.DOCS, pbar)
            cls.__save_index(code_chunks, code_index, DocumentType.CODE, pbar)

    @classmethod
    def __get_chunks_index(cls, chunks: list[MinimalSource], pbar) -> dict:

        tf: dict[str, dict[int, int]] = {}
        idf = {}

        current_path = ""
        file_content = ""
        chunk_content = ""
        total_tokens_count = 0
        chunks_len = len(chunks)

        for chunk in chunks:
            if current_path != chunk.file_path:
                current_path = chunk.file_path
                with open(chunk.file_path) as f:
                    file_content = f.read()

            chunk_content = file_content[
                chunk.first_character_index : chunk.last_character_index + 1
            ]

            tokens = Tokenizer.tokenize(chunk_content)
            chunk.tokens_len = len(tokens)
            tokens_frequency = Counter(tokens)
            total_tokens_count += len(tokens)

            for token, frequency in tokens_frequency.items():

                if token not in tf:
                    tf[token] = {}

                tf[token][chunk.id] = frequency
            pbar.update(1)

        avgdl = total_tokens_count / chunks_len

        for token in tf:
            idf[token] = math.log(
                chunks_len / len(tf[token]),
            )
        return {"tf": tf, "idf": idf, "avgdl": avgdl}

    @classmethod
    def __save_index(
        cls,
        chunks: list[MinimalSource],
        index: dict,
        document_type: DocumentType,
        pbar,
    ):

        if document_type == DocumentType.DOCS:
            avgdl_path = PathInfo.DOCS_AVGDL.value
            tf_path = PathInfo.DOCS_TF.value
            idf_path = PathInfo.DOCS_IDF.value
            chunks_path = PathInfo.DOCS_CHUNKS.value

        else:
            avgdl_path = PathInfo.CODE_AVGDL.value
            tf_path = PathInfo.CODE_TF.value
            idf_path = PathInfo.CODE_IDF.value
            chunks_path = PathInfo.CODE_CHUNKS.value

        with open(avgdl_path, "w") as f:
            json.dump({"avgdl": index["avgdl"]}, f, indent=4)
        pbar.update(1)

        with open(
            tf_path,
            "w",
        ) as f:
            json.dump(index["tf"], f, indent=4)
        pbar.update(1)

        with open(idf_path, "w") as f:
            json.dump(index["idf"], f, indent=4)
        pbar.update(1)

        with open(chunks_path, "w") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4)
        pbar.update(1)

    @classmethod
    def __chunk_docs(cls, max_chunk_size: int, overlap_size: int):
        raw_data_path = PathInfo.RAW_DATA.value
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
        id = 0

        for path in raw_data_path.rglob("*.md"):
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
                            tokens_len=0,
                        )
                    )
                    start_idx += max_chunk_size - overlap_size
                    id += 1
        return chunks_objs

    @classmethod
    def __chunk_code(cls, max_chunk_size: int):
        raw_data_path = PathInfo.RAW_DATA.value
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
        id = 0
        overlap_size = 300  # just for test
        for path in raw_data_path.rglob("*.py"):
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
                            tokens_len=0,
                        )
                    )
                    start_idx += max_chunk_size - overlap_size
                    id += 1

        return chunks_objs
