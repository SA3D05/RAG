import json
import sys
from pydantic import TypeAdapter, ValidationError
from .enums import PathInfo, DocumentType
from .models import MinimalSource, Tokenizer, FileOP


class Retrieval:
    def __init__(self) -> None:
        self.__get_parsed_attrs()

    def __get_parsed_attrs(self):
        current_file = ""
        try:
            current_file = PathInfo.DOCS_CHUNKS.value
            self.docs_chunks = TypeAdapter(list[MinimalSource]).validate_json(
                FileOP.readfile(current_file)
            )

            current_file = PathInfo.DOCS_TF.value
            self.docs_tf = json.loads(FileOP.readfile(current_file))

            current_file = PathInfo.DOCS_IDF.value
            self.docs_idf = json.loads(FileOP.readfile(current_file))

            current_file = PathInfo.DOCS_AVGDL.value
            self.docs_avgdl = json.loads(FileOP.readfile(current_file))[
                "avgdl"
            ]

            current_file = PathInfo.CODE_CHUNKS.value
            self.code_chunks = TypeAdapter(list[MinimalSource]).validate_json(
                FileOP.readfile(current_file)
            )

            current_file = PathInfo.CODE_TF.value
            self.code_tf = json.loads(FileOP.readfile(current_file))

            current_file = PathInfo.CODE_IDF.value
            self.code_idf = json.loads(FileOP.readfile(current_file))

            current_file = PathInfo.CODE_AVGDL.value
            self.code_avgdl = json.loads(FileOP.readfile(current_file))[
                "avgdl"
            ]

        except ValidationError as e:
            print("File:", current_file)
            for error in e.errors():
                msg = error["msg"]
                print(msg)
            sys.exit(1)

        except json.decoder.JSONDecodeError as e:
            print("File:", current_file)
            print(e)
            sys.exit(1)

        except KeyError as e:
            print("File:", current_file)
            print("Error: Expecting field", e)
            sys.exit(1)

    def search(
        self, query: str, k: int, query_type: DocumentType
    ) -> list[MinimalSource]:

        if query_type == DocumentType.DOCS:
            query_tokens = Tokenizer.tokenize(query)
            result = self.__search_docs(query_tokens, k)
            return result

        query_tokens = Tokenizer.tokenize(query)
        result = self.__search_code(query_tokens, k)
        return result

    def __search_docs(
        self, query_tokens: list[str], k: int
    ) -> list[MinimalSource]:
        scores = {}
        result: list[MinimalSource] = []

        for token in query_tokens:

            if token not in self.docs_idf:  # skip unknown tokens in query
                continue

            token_frequency = self.docs_tf[token]
            inverse_document_frequency = self.docs_idf[token]

            for chunk_id, tf_value in token_frequency.items():

                doc_len = self.docs_chunks[int(chunk_id)].tokens_len

                if doc_len is None:
                    print("File:", PathInfo.DOCS_CHUNKS.value)
                    print(
                        "Error: missing 'tokens_len' field in chunk index:",
                        chunk_id,
                    )
                    sys.exit(1)

                chunk_score = self.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    doc_len,
                    self.docs_avgdl,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = self.docs_chunks[int(id)]
            result.append(chunk)

        return result

    def __search_code(
        self, query_tokens: list[str], k: int
    ) -> list[MinimalSource]:
        scores = {}
        result: list[MinimalSource] = []
        for token in query_tokens:

            if token not in self.code_idf:  # skip unknown tokens in query
                continue

            token_frequency = self.code_tf[token]
            inverse_document_frequency = self.code_idf[token]

            for chunk_id, tf_value in token_frequency.items():

                doc_len = self.code_chunks[int(chunk_id)].tokens_len

                if doc_len is None:
                    print("File:", PathInfo.CODE_CHUNKS.value)
                    print(
                        "Error: missing 'tokens_len' field in chunk index:",
                        chunk_id,
                    )
                    sys.exit(1)

                chunk_score = self.__score_chunk(
                    tf_value,
                    inverse_document_frequency,
                    doc_len,
                    self.code_avgdl,
                )

                current_score = 0

                if chunk_id in scores:  # safe check the key
                    current_score = scores[chunk_id]

                scores[chunk_id] = current_score + chunk_score

        top_relevant_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )[:k]

        for id in top_relevant_ids:

            chunk = self.code_chunks[int(id)]
            result.append(chunk)

        return result

    def __score_chunk(
        self,
        tf: int,
        idf: float,
        doc_len: int,
        avgdl: float,
    ) -> float:
        k1 = 1.5
        b = 0.75

        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))

        return idf * (numerator / denominator)
