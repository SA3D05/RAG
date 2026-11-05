import sys
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import uuid
import re


class Tokenizer:
    IDENTIFIER_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        if not text:
            return []

        tokens: list[str] = []

        matches = cls.IDENTIFIER_REGEX.findall(text)

        for match in matches:
            parts = re.split(r"[_\s]+", match)
            for part in parts:
                clean = part.lower().strip()
                if len(clean) > 1 and clean:
                    tokens.append(clean)

        return tokens


class MinimalSource(BaseModel):
    id: int | None = None
    file_path: str
    first_character_index: int
    last_character_index: int
    tokens_len: int | None = None


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class RagDatasetRecall(BaseModel):
    rag_questions: list[AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int


class AvrageDocumentLengthCache(BaseModel):
    docs_avgdl: float
    code_avgdl: float


class ChunksCache(BaseModel):

    docs_chunks: list[MinimalSource]
    code_chunks: list[MinimalSource]


class InverseDocumentFrequencyCache(BaseModel):
    docs_idf: dict[str, float]
    code_idf: dict[str, float]


class TermFrequencyCache(BaseModel):
    docs_tf: dict[str, dict[str, int]]
    code_tf: dict[str, dict[str, int]]


class Recall:
    def __init__(self) -> None:
        self.min_iou = 0.05

    def __get_intersection(
        self, truth_source: MinimalSource, result_source: MinimalSource
    ):

        truth_start = truth_source.first_character_index
        truth_end = truth_source.last_character_index

        result_start = result_source.first_character_index
        result_end = result_source.last_character_index

        return min(result_end, truth_end) - max(result_start, truth_start)

    def __get_union(
        self, truth_source: MinimalSource, result_source: MinimalSource
    ):

        truth_start = truth_source.first_character_index
        truth_end = truth_source.last_character_index

        result_start = result_source.first_character_index
        result_end = result_source.last_character_index

        return max(result_end, truth_end) - min(result_start, truth_start)

    def __check_iou(
        self, truth_source: MinimalSource, result_source: MinimalSource
    ) -> bool:

        intersection = self.__get_intersection(truth_source, result_source)
        union = self.__get_union(truth_source, result_source)

        if intersection / union > self.min_iou:
            return True
        return False

    def __is_found(
        self,
        truth_source: MinimalSource,
        result_source: list[MinimalSource],
    ) -> bool:
        for source in result_source:
            if source.file_path != truth_source.file_path:
                continue
            if self.__check_iou(truth_source, source):
                return True
        return False

    def evaluate(
        self,
        results: list[MinimalSearchResults],
        ground_truth: dict[str, list[MinimalSource]],
    ) -> dict:
        recalls = {
            1: 0.0,
            3: 0.0,
            5: 0.0,
            10: 0.0,
        }
        for question_result in results:
            truth_source = ground_truth[question_result.question_id][0]
            for k in recalls.keys():

                if self.__is_found(
                    truth_source, question_result.retrieved_sources[:k]
                ):
                    recalls[k] = recalls[k] + 1
        results_len = len(results)

        for k, v in recalls.items():
            recalls[k] = v / results_len

        return recalls


class FileOP:

    @staticmethod
    def readfile(file_path: str, pbar: Any = None) -> str:
        try:
            with open(str(file_path)) as f:
                return f.read()
        except OSError as e:
            if pbar is not None:
                pbar.close()
            print("\nFile:", file_path)
            print("Error:", e.strerror)
            sys.exit(1)
        return ""

    @staticmethod
    def writefile(file_path: str, content: str, pbar: Any = None) -> None:
        try:
            with open(file_path, "w") as f:
                f.write(content)
        except OSError as e:
            if pbar is not None:
                pbar.close()
            print("\nFile:", file_path)
            print("Error:", e.strerror)
            sys.exit(1)

    @staticmethod
    def get_pydantic_model(file_path: str, obj: Any) -> Any:
        content = FileOP.readfile(file_path)
        try:
            return obj.model_validate_json(content)

        except ValidationError as e:
            print("File:", file_path)
            print(
                "Error:",
                f"file content must be a valid '{obj.__name__}' Model",
            )
            # for error in e.errors():
            #     msg = error["msg"]
            print(e)

            sys.exit(1)
