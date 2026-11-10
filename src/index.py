import math
from pathlib import Path
import sys
from typing import Any
from tqdm import tqdm
import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree
from .enums import PathInfo, Statics
from .models import (
    AvrageDocumentLengthCache,
    ChunksCache,
    InverseDocumentFrequencyCache,
    MinimalSource,
    TermFrequencyCache,
    Tokenizer,
    Utils,
)


class Indexer:

    def __init__(self, raw_data_path: Path, max_chunk_size: int) -> None:
        self.raw_data_path = raw_data_path
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = Statics.CODE_MIN_CHUNK_SIZE.value
        self.target_files_types = ["py", "txt", "md"]

        self.index_result: dict[str, Any] = {
            "docs_tf": {},
            "docs_idf": {},
            "docs_avgdl": 0.0,
            "code_tf": {},
            "code_idf": {},
            "code_avgdl": 0.0,
        }

        self.code_chunks: list[MinimalSource] = []
        self.docs_chunks: list[MinimalSource] = []

        self.target_paths = []

        self.__set_target_paths()

    def index(self) -> None:

        self.__check_files()

        Utils.write_log("chunking ...")
        with tqdm(total=len(self.target_paths), file=sys.stdout) as pbar:
            self.__chunk(pbar)

        docs_chunks_len = len(self.docs_chunks)
        code_chunks_len = len(self.code_chunks)

        if docs_chunks_len + code_chunks_len == 0:
            Utils.write_log("Nothing to index!")
            sys.exit()

        Utils.write_log("indexing ...")
        with tqdm(
            total=docs_chunks_len + code_chunks_len, file=sys.stdout
        ) as pbar:
            self.__set_index_from_chunks(pbar)

        Utils.write_log("saving ...")
        with tqdm(total=4, file=sys.stdout) as pbar:
            self.__save_index(pbar)

    def __set_target_paths(self):

        for file_pattern in self.target_files_types:
            self.target_paths.extend(
                [p for p in self.raw_data_path.rglob("*." + file_pattern)]
            )

    def __check_files(self):
        file_path = ""

        if not self.raw_data_path.is_dir():
            Utils.write_err(f"Error: '{self.raw_data_path}' not found")
            sys.exit(1)
        try:

            for p in PathInfo:
                Utils.get_path(p.value, create=True)

            self.__save_index()
        except OSError as e:

            Utils.write_err(f"File: {file_path}")
            Utils.write_err(f"Error: {e.strerror}")
            sys.exit(1)

        if not self.target_paths:
            Utils.write_log("Nothing to chunk!")
            sys.exit()

    def __get_tokens_frequency(self, tokens: list[str]) -> dict[str, int]:
        tokens_frequency: dict[str, int] = {}
        for token in tokens:
            if token in tokens_frequency:
                continue
            tokens_frequency[token] = tokens.count(token)
        return tokens_frequency

    def __set_index_from_chunks(self, pbar=None) -> None:

        for chunk_type in ("docs", "code"):

            chunks: list[MinimalSource] = getattr(self, chunk_type + "_chunks")

            if not chunks:
                continue

            term_frequency: dict[str, dict[str, int]] = {}
            inverse_document_frequency = {}

            current_path = ""
            file_content = ""
            chunk_content = ""
            total_tokens_count = 0
            chunks_len = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):

                if current_path != chunk.file_path:
                    current_path = chunk.file_path

                    with open(chunk.file_path) as f:
                        file_content = f.read()

                chunk_content = file_content[
                    chunk.first_character_index : chunk.last_character_index
                    + 1
                ]

                tokens = Tokenizer.tokenize(chunk_content + " " + current_path)
                tokens_frequency = self.__get_tokens_frequency(tokens)
                chunk.tokens_len = len(tokens)
                total_tokens_count += chunk.tokens_len

                for token, frequency in tokens_frequency.items():

                    term_frequency.setdefault(token, {})

                    term_frequency[token][str(chunk_idx)] = frequency

                if pbar is not None:
                    pbar.update(1)

            avgdl = total_tokens_count / chunks_len

            for token in term_frequency:
                inverse_document_frequency[token] = math.log(
                    chunks_len / len(term_frequency[token])
                )

            self.index_result.update(
                {
                    f"{chunk_type}_tf": term_frequency,
                    f"{chunk_type}_idf": inverse_document_frequency,
                    f"{chunk_type}_avgdl": avgdl,
                }
            )

    def __save_index(self, pbar=None):

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
            docs_chunks=[] if not self.docs_chunks else self.docs_chunks,
            code_chunks=[] if not self.code_chunks else self.code_chunks,
        )

        for obj, path in zip([chunks, tf, idf, avgdl], PathInfo):
            Utils.writefile(
                str(path.value),
                obj.model_dump_json(indent=4),
                pbar,
            )
            if pbar is not None:
                pbar.update(1)

    def __chunk_docs(self, content, path):
        chunks_objs: list[MinimalSource] = []

        start_idx = 0
        end_idx = 0
        content_len = len(content)
        while start_idx < content_len:
            end_idx = min(start_idx + self.max_chunk_size, content_len) - 1
            chunks_objs.append(
                MinimalSource(
                    file_path=str(path),
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                    tokens_len=0,
                )
            )
            start_idx += self.max_chunk_size - Statics.DOCS_OVERLAP_SIZE.value

        return chunks_objs

    def __chunk_code(self, tree: Tree, byte_map, path) -> list[MinimalSource]:
        chunks_objs: list[MinimalSource] = []
        buffer_start_byte = -1

        def flush_buffer(end_byte: int):
            nonlocal buffer_start_byte
            if buffer_start_byte != -1:
                chunks_objs.append(
                    MinimalSource(
                        file_path=str(path),
                        first_character_index=byte_map[buffer_start_byte],
                        last_character_index=byte_map[end_byte],
                        tokens_len=0,
                    )
                )
                buffer_start_byte = -1

        def rec(root: Node):
            nonlocal buffer_start_byte
            for child in root.children:
                byte_range = child.end_byte - child.start_byte

                if byte_range > self.max_chunk_size:
                    flush_buffer(child.start_byte)
                    rec(child)

                elif byte_range < self.min_chunk_size:
                    if buffer_start_byte == -1:
                        buffer_start_byte = child.start_byte

                    elif (
                        child.end_byte - buffer_start_byte
                    ) > self.min_chunk_size:
                        flush_buffer(child.end_byte)

                else:
                    flush_buffer(child.start_byte)

                    chunks_objs.append(
                        MinimalSource(
                            file_path=str(path),
                            first_character_index=byte_map[child.start_byte],
                            last_character_index=byte_map[child.end_byte],
                            tokens_len=0,
                        )
                    )

        rec(tree.root_node)

        if buffer_start_byte != -1:
            flush_buffer(tree.root_node.end_byte)

        return chunks_objs

    def __get_byte_map(self, content: str):
        result = {}
        byte_idx = 0
        for character_idx, character in enumerate(content):
            result[byte_idx] = character_idx
            byte_idx += len(character.encode())
        result[byte_idx] = len(content)
        return result

    def __chunk(self, pbar=None) -> None:

        PY_LANGUAGE = Language(tree_sitter_python.language())
        parser = Parser(PY_LANGUAGE)
        for path in self.target_paths:
            with open(path) as f:
                content = f.read()
                if not content:
                    if pbar is not None:
                        pbar.update(1)
                    continue

                if path.suffix == ".py":
                    # chunking code
                    encoded_content = content.encode()
                    byte_map = self.__get_byte_map(content)
                    tree = parser.parse(encoded_content)
                    current_chunks = self.__chunk_code(tree, byte_map, path)
                    self.code_chunks.extend(current_chunks)

                else:
                    # chunking docs
                    current_chunks = self.__chunk_docs(content, path)
                    self.docs_chunks.extend(current_chunks)

            if pbar is not None:
                pbar.update(1)
