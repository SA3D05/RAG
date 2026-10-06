from pydantic import BaseModel, Field
import uuid
import re

# 1. Regex to split PascalCase / camelCase
# Example: "AsyncLLMEngine" -> splits between lower/upper, acronym/title
CAMEL_SPLIT_REGEX = re.compile(r"([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z])")

# 2. Extract valid Python identifiers (including leading/trailing underscores)
IDENTIFIER_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# 3. Programming and natural language stopwords to exclude from BM25
PYTHON_CODE_STOPWORDS = {
    # Python keywords & common boilerplate
    "def",
    "class",
    "return",
    "self",
    "cls",
    "import",
    "from",
    "as",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "in",
    "is",
    "not",
    "and",
    "or",
    "try",
    "except",
    "finally",
    "with",
    "pass",
    "raise",
    "yield",
    "true",
    "false",
    "none",
    "lambda",
    "global",
    "nonlocal",
    "assert",
    # English fluff words common in docstrings/comments
    "the",
    "a",
    "an",
    "this",
    "that",
    "it",
    "to",
    "of",
    "for",
    "on",
    "at",
    "by",
}


def _split_identifier(name: str) -> list[str]:
    """Expands an identifier into full form and sub-tokens.

    Examples:
        'AsyncLLMEngine'     -> ['asyncllmengine', 'async', 'llm', 'engine']
        'init_cache_engine'  -> ['init_cache_engine', 'init', 'cache', 'engine']
        '__init__'           -> ['__init__', 'init']
    """
    sub_tokens: list[str] = []
    lower_name = name.lower()

    # Keep full identifier (critical for exact symbol matches)
    if len(name) > 1 and lower_name not in PYTHON_CODE_STOPWORDS:
        sub_tokens.append(lower_name)

    # Split camelCase / PascalCase
    # e.g., "AsyncLLMEngine" -> "Async LLM Engine"
    spaced_camel = CAMEL_SPLIT_REGEX.sub(
        lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}",
        name,
    )

    # Split on whitespace and underscores for atomic sub-tokens
    parts = re.split(r"[_\s]+", spaced_camel)
    for part in parts:
        clean = part.lower().strip()
        if len(clean) > 1 and clean not in PYTHON_CODE_STOPWORDS:
            sub_tokens.append(clean)

    return sub_tokens


def tokenize_python_code(text: str) -> list[str]:
    """Tokenize Python code chunks and code queries for BM25 indexing."""
    if not text:
        return []

    tokens: list[str] = []

    # Find all Python identifiers across the code
    matches = IDENTIFIER_REGEX.findall(text)
    for match in matches:
        extracted = _split_identifier(match)
        tokens.extend(extracted)

    return tokens


class Tokenizer:

    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Tokenize text for BM25 indexing, preserving term frequency."""
        if not text:
            return []

        tokens: list[str] = []

        # Find all clean words, stripping punctuation, markdown markers, parens
        for match in cls.TOKEN_PATTERN.finditer(text.lower()):
            word = match.group(0)

            # Strip leading/trailing dashes or underscores
            word = word.strip("-_")

            if len(word) > 1:
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
