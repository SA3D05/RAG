from pydantic import BaseModel, Field
from typing import List
import uuid
import re


class Tokenizer:

    @staticmethod
    def tokenize(text: str) -> set[str]:
        """Placeholder method for demonstrait tokenizing"""
        if not text:
            return set()

        # [\W_]+ ahmed suggestion
        text = text.lower()
        # 1. Strip code block backticks, inline backticks, and HTML tags
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`.*?`", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)

        # 2. Strip URLs and markdown link syntax [text](url) -> keep 'text'
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 3. Strip headers (#), bold/italic (*, _), blockquotes (>), list markers
        text = re.sub(r"[#*_~>`\-+=|]", " ", text)

        # 4. Split camelCase/PascalCase identifiers into separate terms
        # e.g., 'AsyncLLMEngine' -> 'Async LLM Engine'
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

        # 5. Extract words and code identifiers, stripping snake_case/punctuation
        # Matches alphanumeric sequences of length >= 2
        raw_tokens: list[str] = re.findall(r"[a-zA-Z0-9]+", text.lower())

        # Filter out single characters or standalone pure digits if desired
        return {token for token in raw_tokens if len(token) > 1}


class MinimalSource(BaseModel):
    id: int
    file_path: str
    # text: str
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
