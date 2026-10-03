import sys

from pydantic import BaseModel, Field
from typing import List
import uuid
import re


class Tokenizer:

    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")

    STOPWORDS = {
        "a",
        "you",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize text for BM25 indexing, preserving term frequency."""
        if not text:
            return []

        tokens: List[str] = []

        # Find all clean words, stripping punctuation, markdown markers, parens
        for match in cls.TOKEN_PATTERN.finditer(text.lower()):
            word = match.group(0)

            # Strip leading/trailing dashes or underscores
            word = word.strip("-_")

            if len(word) > 1 and word not in cls.STOPWORDS:
                tokens.append(word)
        return tokens


class MinimalSource(BaseModel):
    id: int
    file_path: str
    first_character_index: int
    last_character_index: int
    tokens_len: int


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
