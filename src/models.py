from pydantic import BaseModel, Field
import uuid
import re


class Tokenizer:
    # 1. Regex to split PascalCase / camelCase
    # Example: "AsyncLLMEngine" -> splits between lower/upper, acronym/title
    CAMEL_SPLIT_REGEX = re.compile(r"([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z])")

    # 2. Extract valid Python identifiers (including leading/trailing underscores)
    IDENTIFIER_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")

    @classmethod
    def __split_identifier(cls, name: str) -> list[str]:

        sub_tokens: list[str] = []
        lower_name = name.lower()

        # Keep full identifier (critical for exact symbol matches)
        if len(name) > 1 and lower_name:
            sub_tokens.append(lower_name)

        ## Split camelCase / PascalCase
        # # e.g., "AsyncLLMEngine" -> "Async LLM Engine"
        spaced_camel = cls.CAMEL_SPLIT_REGEX.sub(
            lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}",
            name,
        )

        # # Split on whitespace and underscores for atomic sub-tokens
        parts = re.split(r"[_\s]+", lower_name)
        for part in parts:
            clean = part.lower().strip()
            if len(clean) > 1 and clean:
                sub_tokens.append(clean)

        return sub_tokens

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        if not text:
            return []

        tokens: list[str] = []

        matches = cls.IDENTIFIER_REGEX.findall(text)

        for match in matches:
            extracted = cls.__split_identifier(match)
            tokens.extend(extracted)
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
