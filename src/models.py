from pydantic import BaseModel, Field
from typing import List
import uuid
import re


class Tokenizer:

    @staticmethod
    def tokenize(text: str) -> set[str]:
        """Tokenize Markdown text for RAG indexing."""

        if not text:
            return set()

        text = text.lower()
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\n?", " ", text)
        text = text.replace("```", " ")

        text = text.replace("`", " ")

        text = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            text,
        )

        text = re.sub(
            r"\[([^\]]+)\]\[[^\]]*\]",
            r"\1",
            text,
        )

        text = re.sub(
            r"(?:https?://|www\.)\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"<[^>]+>", " ", text)

        text = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            text,
        )

        text = re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1 \2",
            text,
        )

        text = re.sub(r"[_\-./\\]+", " ", text)

        text = re.sub(r"[+#]", " ", text)

        text = re.sub(
            r"[#*_~>|=]+",
            " ",
            text,
        )

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
            flags=re.UNICODE,
        )

        raw_tokens = re.findall(
            r"\b[a-z0-9]+\b",
            text,
        )

        return {token for token in raw_tokens if len(token) > 1}


class MinimalSource(BaseModel):
    id: int
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
