import math
from pathlib import Path
import sys
from typing import Any
from tqdm import tqdm
import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree
from .enums import PathInfo, Statics
from .utils import Utils
from .models import (
    AvrageDocumentLengthCache,
    ChunksCache,
    InverseDocumentFrequencyCache,
    MinimalSource,
    TermFrequencyCache,
)


class Indexer:

    def __init__(self, raw_data_path: Path, max_chunk_size: int) -> None:
        self.raw_data_path = raw_data_path
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = Statics.MIN_CHUNK_SIZE.value
        self.target_files_types = ["py", "txt", "md"]

        self.index_result: dict[str, Any] = {"tf": {}, "idf": {}, "avgdl": 0.0}

        self.chunks: list[MinimalSource] = []

        self.target_paths = []

        self.__set_target_paths()

    def index(self) -> None:

        self.__check_files()

        Utils.write_log("chunking ...")
        with tqdm(total=len(self.target_paths), file=sys.stdout) as pbar:
            self.__chunk(pbar)

        chunks_len = len(self.chunks)

        if chunks_len == 0:
            Utils.write_log("Nothing to index!")
            sys.exit()

        Utils.write_log("indexing ...")
        with tqdm(total=chunks_len, file=sys.stdout) as pbar:
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

    def __update_term_frequency(
        self,
        chunk_idx: int,
        chunk_tokens_frequency: dict[str, int],
        term_frequency: dict[str, dict[str, int]],
    ) -> None:

        for token, frequency in chunk_tokens_frequency.items():

            term_frequency.setdefault(token, {})
            term_frequency[token][str(chunk_idx)] = frequency

    def __update_inverse_document_frequency(
        self,
        term_frequency: dict[str, dict[str, int]],
        inverse_document_frequency: dict[str, float],
        chunks_len: int,
    ) -> None:

        for token, token_frequency in term_frequency.items():
            inverse_document_frequency[token] = math.log2(
                chunks_len / len(token_frequency)
            )

    def __set_index_from_chunks(self, pbar=None) -> None:

        term_frequency: dict[str, dict[str, int]] = {}
        inverse_document_frequency: dict[str, float] = {}

        current_path = ""
        file_content = ""
        total_tokens_count = 0
        chunks_len = len(self.chunks)

        for chunk_idx, chunk in enumerate(self.chunks):

            if current_path != chunk.file_path:
                current_path = chunk.file_path
                file_content = Utils.readfile(Path(chunk.file_path), pbar)
            first, last = (
                chunk.first_character_index,
                chunk.last_character_index + 1,
            )
            chunk_content = file_content[first:last]

            tokens = Utils.tokenize(chunk_content + " " + current_path)
            tokens_frequency = self.__get_tokens_frequency(tokens)

            chunk.tokens_len = len(tokens)
            total_tokens_count += chunk.tokens_len

            self.__update_term_frequency(
                chunk_idx, tokens_frequency, term_frequency
            )
            if pbar is not None:
                pbar.update(1)

        avgdl = total_tokens_count / chunks_len
        self.__update_inverse_document_frequency(
            term_frequency, inverse_document_frequency, chunks_len
        )
        self.index_result.update(
            {
                "tf": term_frequency,
                "idf": inverse_document_frequency,
                "avgdl": avgdl,
            }
        )

    def __save_index(self, pbar: tqdm | None = None):

        avgdl = AvrageDocumentLengthCache(avgdl=self.index_result["avgdl"])
        tf = TermFrequencyCache(tf=self.index_result["tf"])
        idf = InverseDocumentFrequencyCache(idf=self.index_result["idf"])
        chunks = ChunksCache(chunks=[] if not self.chunks else self.chunks)

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
            if end_idx == content_len - 1:
                break
            start_idx += self.max_chunk_size - Statics.OVERLAP_SIZE.value - 1

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
                        last_character_index=byte_map[end_byte] - 1,
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
                            last_character_index=byte_map[child.end_byte] - 1,
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

    def __chunk(self, pbar: tqdm | None = None) -> None:

        PY_LANGUAGE = Language(tree_sitter_python.language())
        parser = Parser(PY_LANGUAGE)
        for path in self.target_paths:
            content = Utils.readfile(path, pbar)
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
                self.chunks.extend(current_chunks)

            else:
                # chunking docs
                current_chunks = self.__chunk_docs(content, path)
                self.chunks.extend(current_chunks)

            if pbar is not None:
                pbar.update(1)
