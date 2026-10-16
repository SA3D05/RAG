from pydantic import BaseModel, Field
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
