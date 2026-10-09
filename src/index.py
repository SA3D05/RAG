import ast
import json
import math

from tqdm import tqdm
import tree_sitter_python
from collections import Counter
from tree_sitter import Language, Parser

from .enums import DocumentType, PathInfo
from .models import MinimalSource, Tokenizer


class Indexer:

    @classmethod
    def index(cls, max_chunk_size: int) -> None:

        print("chunking ...")
        code_paths = [p for p in PathInfo.RAW_DATA.value.rglob("*.py")]
        docs_paths = [p for p in PathInfo.RAW_DATA.value.rglob("*.md")]

        with tqdm(total=len(docs_paths) + len(code_paths)) as pbar:
            code_chunks, docs_chunks = cls.__chunk(
                max_chunk_size, docs_paths + code_paths, pbar
            )

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

            tokens = Tokenizer.tokenize(chunk_content + " " + current_path)
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
    def __chunk_docs(
        cls, content, path, max_chunk_size: int, overlap_size: int, id
    ):
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
            start_idx += max_chunk_size - overlap_size
            id += 1
        return chunks_objs

    @classmethod
    def __chunk_code(
        cls, content, path, max_chunk_size: int, id_counter, parser
    ):
        chunks_objs: list[MinimalSource] = []
        tree = parser.parse(content.encode())

        # Accumulators to track the bounds of grouped small nodes
        current_start_byte = -1
        current_end_byte = -1

        for node in tree.root_node.children:
            node_start, node_end = node.byte_range
            node_size = node_end - node_start

            # CASE 1: The node itself is massive (e.g., a giant class/function)
            if node_size > max_chunk_size:
                # Flush the current accumulator before processing the massive node
                if current_start_byte != -1:
                    chunks_objs.append(
                        MinimalSource(
                            id=id_counter,
                            file_path=str(path),
                            first_character_index=current_start_byte,
                            last_character_index=current_end_byte,
                            tokens_len=0,
                        )
                    )
                    id_counter += 1
                    current_start_byte = -1
                    current_end_byte = -1

                # Blindly split the huge node by max_chunk_size
                a = node_start
                buffer = node_size
                while buffer > max_chunk_size:
                    chunks_objs.append(
                        MinimalSource(
                            id=id_counter,
                            file_path=str(path),
                            first_character_index=a,
                            last_character_index=a + max_chunk_size,
                            tokens_len=0,
                        )
                    )
                    a += max_chunk_size
                    buffer -= max_chunk_size
                    id_counter += 1

                if buffer > 0:
                    chunks_objs.append(
                        MinimalSource(
                            id=id_counter,
                            file_path=str(path),
                            first_character_index=a,
                            last_character_index=node_end,
                            tokens_len=0,
                        )
                    )
                    id_counter += 1

            # CASE 2: Normal/Small nodes (imports, small functions)
            else:
                if current_start_byte == -1:
                    # Start a new grouping buffer
                    current_start_byte = node_start
                    current_end_byte = node_end

                elif (node_end - current_start_byte) <= max_chunk_size:
                    # The node fits in our current grouping; extend the boundaries
                    current_end_byte = node_end

                else:
                    # Adding this node would exceed max_chunk_size; flush current grouping
                    chunks_objs.append(
                        MinimalSource(
                            id=id_counter,
                            file_path=str(path),
                            first_character_index=current_start_byte,
                            last_character_index=current_end_byte,
                            tokens_len=0,
                        )
                    )
                    id_counter += 1
                    # Start a new grouping with the current node
                    current_start_byte = node_start
                    current_end_byte = node_end

        # Clean up: flush whatever is left in the grouping buffer at the end of the file
        if current_start_byte != -1:
            chunks_objs.append(
                MinimalSource(
                    id=id_counter,
                    file_path=str(path),
                    first_character_index=current_start_byte,
                    last_character_index=current_end_byte,
                    tokens_len=0,
                )
            )
            id_counter += 1

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
                    current_chunks = cls.__chunk_code(
                        content, path, max_chunk_size, code_id, parser
                    )
                    code_id += len(current_chunks)
                    code_chunks_objs.extend(current_chunks)

                except SyntaxError:
                    current_chunks = cls.__chunk_docs(
                        content, path, max_chunk_size, 300, docs_id
                    )
                    docs_id += len(current_chunks)
                    docs_chunks_objs.extend(current_chunks)

            pbar.update(1)

        return (code_chunks_objs, docs_chunks_objs)
