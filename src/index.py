import json
import math
from pathlib import Path
from pprint import pprint
from .models import MinimalSource, Tokenizer
from collections import Counter


class Indexer:

    @classmethod
    def index(
        cls, max_chunk_size: int = 2000, overlap_size: int = 100
    ) -> None:
        chunks = cls.__chunk(max_chunk_size, overlap_size)

        tfs: dict[str, dict] = {}
        idfs = {}
        current_path = ""
        file_content = ""
        chunk_content = ""

        for chunk in chunks:
            if current_path != chunk.file_path:
                current_path = chunk.file_path
                with open(chunk.file_path) as f:
                    file_content = f.read()

            chunk_content = file_content[
                chunk.first_character_index : chunk.last_character_index + 1
            ].lower()
            tokens = Tokenizer.tokenize(chunk_content)

            tf_counts = Counter(tokens)

            for token, count in tf_counts.items():
                if token not in tfs:
                    tfs[token] = {}
                tfs[token][chunk.id] = count
        chunks_len = len(chunks)

        for token in tfs:
            idfs[token] = math.log(chunks_len / len(tfs[token]))

        with open("data/processed/tf.json", "w") as f:
            json.dump(tfs, f, indent=4)

        with open("data/processed/idf.json", "w") as f:
            json.dump(idfs, f, indent=4)

        with open("data/processed/chunks.json", "w") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4)

    @classmethod
    def __chunk(cls, max_chunk_size: int, overlap_size: int):
        base_path = Path("data/raw/vllm-0.10.1")
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
