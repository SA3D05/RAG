import ast
import json
import math
from pathlib import Path
import sys

from tqdm import tqdm
import tree_sitter_python
from collections import Counter
from tree_sitter import Language, Parser

from .enums import DocumentType, PathInfo, Statics
from .models import MinimalSource, Tokenizer, FileOP


class Indexer:

    @classmethod
    def index(cls, max_chunk_size: int) -> None:
        print("chunking ...")
        raw_data = Path("data/raw")
        cls.__check_files(raw_data)

        code_paths = [p for p in raw_data.rglob("*.py")]
        docs_paths = [p for p in raw_data.rglob("*.md")]
        docs_paths.extend([p for p in raw_data.rglob("*.txt")])

        if not any([code_paths, docs_paths]):
            print("Nothing to chunk!")
            return
        with tqdm(total=len(docs_paths) + len(code_paths)) as pbar:
            code_chunks, docs_chunks = cls.__chunk(
                max_chunk_size, docs_paths + code_paths, pbar
            )

        docs_index = None
        code_index = None
        print("indexing ...")
        with tqdm(total=len(code_chunks) + len(docs_chunks)) as pbar:
            docs_index = cls.__get_chunks_index(docs_chunks, pbar)
            code_index = cls.__get_chunks_index(code_chunks, pbar)

        print("saving ...")
        with tqdm(total=8) as pbar:
            cls.__save_index(
                docs_chunks,
                code_chunks,
                docs_index,
                code_index,
                pbar,
            )

    @classmethod
    def __check_files(cls, raw_data: Path):
        file_path = ""

        if not raw_data.is_dir():
            print(f"Error: '{raw_data}' not found")
            sys.exit(1)
        try:
            for p in PathInfo:
                file_path = p.value
                with open(file_path, "w") as f:
                    pass
        except OSError as e:
            print("File:", file_path)
            print("Error:", e.strerror)
            sys.exit(1)

    @classmethod
    def __get_chunks_index(cls, chunks: list[MinimalSource], pbar) -> dict:
        tf: dict[str, dict[str, int]] = {}
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

            tokens = Tokenizer.tokenize(chunk_content + " " + current_path)
            chunk.tokens_len = len(tokens)
            tokens_frequency = Counter(tokens)
            total_tokens_count += len(tokens)

            for token, frequency in tokens_frequency.items():

                tf.setdefault(token, {})

                tf[token][str(chunk.id)] = frequency

            pbar.update(1)

        avgdl = total_tokens_count / chunks_len

        for token in tf:
            idf[token] = math.log(chunks_len / len(tf[token]))

        return {"tf": tf, "idf": idf, "avgdl": avgdl}

    @classmethod
    def __save_index(
        cls,
        docs_chunks: list[MinimalSource],
        code_chunks: list[MinimalSource],
        docs_index: dict,
        code_index: dict,
        pbar,
    ):

        avgdl_path = PathInfo.AVGDL_PATH.value
        tf_path = PathInfo.TF_PATH.value
        idf_path = PathInfo.IDF_PATH.value
        chunks_path = PathInfo.CHUNKS_PATH.value

        FileOP.writefile(
            str(avgdl_path),
            json.dumps(
                {
                    "docs_avgdl": docs_index["avgdl"],
                    "code_avgdl": code_index["avgdl"],
                },
                indent=4,
            ),
            pbar,
        )
        pbar.update(1)

        FileOP.writefile(
            str(tf_path),
            json.dumps(
                {"docs_tf": docs_index["tf"], "code_tf": code_index["tf"]},
                indent=4,
            ),
            pbar,
        )
        pbar.update(1)

        FileOP.writefile(
            str(idf_path),
            json.dumps(
                {"docs_idf": docs_index["idf"], "code_idf": code_index["idf"]},
                indent=4,
            ),
            pbar,
        )
        pbar.update(1)

        FileOP.writefile(
            str(chunks_path),
            json.dumps(
                {
                    "docs_chunks": [
                        chunk.model_dump() for chunk in docs_chunks
                    ],
                    "code_chunks": [
                        chunk.model_dump() for chunk in code_chunks
                    ],
                },
                indent=4,
            ),
            pbar,
        )
        pbar.update(1)

    @classmethod
    def __chunk_docs(cls, content, path, max_chunk_size: int, id):
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
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
            start_idx += max_chunk_size - Statics.DOCS_OVERLAP_SIZE.value
            id += 1

        return chunks_objs

    @classmethod
    def __chunk_code(
        cls, tree, max_chunk_size, id_counter, path
    ) -> list[MinimalSource]:
        chunks_objs: list[MinimalSource] = []
        start_byte = -1
        end_byte = -1
        range_size = -1

        def add_chunk():
            nonlocal id_counter
            chunks_objs.append(
                MinimalSource(
                    id=id_counter,
                    file_path=str(path),
                    first_character_index=start_byte,
                    last_character_index=start_byte + max_chunk_size,
                    tokens_len=0,
                )
            )
            id_counter += 1

        def size_big():
            nonlocal start_byte
            nonlocal end_byte
            nonlocal range_size
            nonlocal id_counter

            add_chunk()
            start_byte += max_chunk_size
            range_size -= max_chunk_size

        for child in tree.root_node.children:

            if start_byte != -1:
                _, end_byte = child.byte_range
            else:
                start_byte, end_byte = child.byte_range

            range_size = end_byte - start_byte

            while range_size > max_chunk_size:
                size_big()
            if range_size < Statics.CODE_MIN_CHUNK_SIZE.value:
                continue
            add_chunk()
            start_byte = -1
            end_byte = -1

        if start_byte != -1 or end_byte != -1:

            range_size = end_byte - start_byte

            while range_size > max_chunk_size:
                size_big()
            if range_size > 0:
                add_chunk()
        return chunks_objs

    @classmethod
    def __chunk(
        cls, max_chunk_size: int, paths, pbar
    ) -> tuple[list[MinimalSource], list[MinimalSource]]:

        code_chunks_objs: list[MinimalSource] = []
        docs_chunks_objs: list[MinimalSource] = []
        PY_LANGUAGE = Language(tree_sitter_python.language())
        parser = Parser(PY_LANGUAGE)
        code_id = 0
        docs_id = 0
        for path in paths:
            with open(path) as f:
                content = f.read()
                try:
                    ast.parse(content)
                    tree = parser.parse(content.encode())
                    current_chunks = cls.__chunk_code(
                        tree, max_chunk_size, code_id, path
                    )
                    code_id += len(current_chunks)
                    code_chunks_objs.extend(current_chunks)

                except SyntaxError:
                    current_chunks = cls.__chunk_docs(
                        content, path, max_chunk_size, docs_id
                    )
                    docs_id += len(current_chunks)
                    docs_chunks_objs.extend(current_chunks)

            pbar.update(1)

        return (code_chunks_objs, docs_chunks_objs)
