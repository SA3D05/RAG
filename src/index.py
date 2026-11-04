import math
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm
import tree_sitter_python
from collections import Counter
from tree_sitter import Language, Parser

from .enums import PathInfo, Statics
from .models import (
    AvrageDocumentLengthCache,
    ChunksCache,
    InverseDocumentFrequencyCache,
    MinimalSource,
    TermFrequencyCache,
    Tokenizer,
    FileOP,
)


class Indexer:

    def __init__(self, raw_data_path: Path, max_chunk_size: int) -> None:
        self.raw_data_path = raw_data_path
        self.max_chunk_size = max_chunk_size

        self.target_files_types = ["py", "txt", "md"]

        self.index_result: dict[str, Any] = {
            "docs_tf": None,
            "docs_idf": None,
            "docs_avgdl": None,
            "code_tf": None,
            "code_idf": None,
            "code_avgdl": None,
        }

        self.code_chunks: list[MinimalSource] = []
        self.docs_chunks: list[MinimalSource] = []

        self.target_paths = []

        self.__set_target_paths()

    def index(self) -> None:

        self.__check_files()

        self.__chunk()

        self.__set_index_from_chunks()

        self.__save_index()

    def __set_target_paths(self):

        for file_pattern in self.target_files_types:
            self.target_paths.extend(
                [p for p in self.raw_data_path.rglob("*." + file_pattern)]
            )

    def __check_files(self):
        file_path = ""

        if not self.raw_data_path.is_dir():
            print(f"Error: '{self.raw_data_path}' not found")
            sys.exit(1)
        try:
            for p in PathInfo:
                file_path = p.value
                with open(file_path, "w") as f:
                    f.write("{}")
        except OSError as e:
            print("File:", file_path)
            print("Error:", e.strerror)
            sys.exit(1)

        if not self.target_paths:
            print("Nothing to chunk!")
            sys.exit()

    def __set_index_from_chunks(self) -> None:

        print("indexing ...")
        with tqdm(total=len(self.code_chunks) + len(self.docs_chunks)) as pbar:
            for chunk_type in ("docs", "code"):
                chunks: list[MinimalSource] = getattr(
                    self, chunk_type + "_chunks"
                )
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
                        chunk.first_character_index : chunk.last_character_index
                        + 1
                    ]

                    tokens = Tokenizer.tokenize(
                        chunk_content + " " + current_path
                    )
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

                self.index_result.update(
                    {
                        f"{chunk_type}_tf": tf,
                        f"{chunk_type}_idf": idf,
                        f"{chunk_type}_avgdl": avgdl,
                    }
                )

    def __save_index(self):

        print("saving ...")
        with tqdm(total=4) as pbar:
            avgdl = AvrageDocumentLengthCache(
                docs_avgdl=self.index_result["docs_avgdl"],
                code_avgdl=self.index_result["code_avgdl"],
            )
            tf = TermFrequencyCache(
                docs_tf=self.index_result["docs_tf"],
                code_tf=self.index_result["code_tf"],
            )

            idf = InverseDocumentFrequencyCache(
                docs_idf=self.index_result["docs_idf"],
                code_idf=self.index_result["code_idf"],
            )
            chunks = ChunksCache(
                docs_chunks=None if not self.docs_chunks else self.docs_chunks,
                code_chunks=None if not self.code_chunks else self.code_chunks,
            )

            for obj, path in zip([chunks, tf, idf, avgdl], PathInfo):
                FileOP.writefile(
                    str(path.value),
                    obj.model_dump_json(indent=4),
                    pbar,
                )
                pbar.update(1)

    def __chunk_docs(self, content, path, id):
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
        content_len = len(content)
        start_idx = 0
        end_idx = 0

        while start_idx <= content_len:
            end_idx = min(start_idx + self.max_chunk_size, content_len) - 1
            chunks_objs.append(
                MinimalSource(
                    id=id,
                    file_path=str(path),
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                    tokens_len=0,
                )
            )
            start_idx += self.max_chunk_size - Statics.DOCS_OVERLAP_SIZE.value
            id += 1

        return chunks_objs

    def __chunk_code(self, tree, id_counter, path) -> list[MinimalSource]:
        chunks_objs: list[MinimalSource] = []
        start_byte = -1
        end_byte = -1
        range_size = -1

        def add_chunk(is_last: bool = False):
            nonlocal id_counter
            nonlocal end_byte
            chunks_objs.append(
                MinimalSource(
                    id=id_counter,
                    file_path=str(path),
                    first_character_index=start_byte,
                    last_character_index=(
                        end_byte - 1
                        if is_last
                        else start_byte + self.max_chunk_size
                    ),
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
            start_byte += self.max_chunk_size
            range_size -= self.max_chunk_size

        for child in tree.root_node.children:

            if start_byte != -1:
                _, end_byte = child.byte_range
            else:
                start_byte, end_byte = child.byte_range

            range_size = end_byte - start_byte

            while range_size > self.max_chunk_size:
                size_big()
            if range_size < Statics.CODE_MIN_CHUNK_SIZE.value:
                continue
            add_chunk()
            start_byte = -1
            end_byte = -1

        if start_byte != -1 or end_byte != -1:

            range_size = end_byte - start_byte

            while range_size > self.max_chunk_size:
                size_big()
            if range_size > 0:
                add_chunk(True)
        return chunks_objs

    def __chunk(self) -> None:

        print("chunking ...")

        PY_LANGUAGE = Language(tree_sitter_python.language())
        parser = Parser(PY_LANGUAGE)
        code_id = 0
        docs_id = 0

        with tqdm(total=len(self.target_paths)) as pbar:

            for path in self.target_paths:
                with open(path) as f:
                    content = f.read()
                    if not content:
                        pbar.update(1)
                        continue

                    if path.suffix != ".py":
                        # chunking docs
                        tree = parser.parse(content.encode())
                        current_chunks = self.__chunk_code(tree, code_id, path)
                        code_id += len(current_chunks)
                        self.docs_chunks.extend(current_chunks)

                    else:
                        # chunking code
                        current_chunks = self.__chunk_docs(
                            content, path, docs_id
                        )
                        docs_id += len(current_chunks)
                        self.code_chunks.extend(current_chunks)

                pbar.update(1)
