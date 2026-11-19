from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Identify a character range and token count in a source document."""

    file_path: str
    first_character_index: int
    last_character_index: int
    tokens_len: int = 0


class UnansweredQuestion(BaseModel):
    """Represent a dataset question that has not been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question together with its answer and source."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset containing answered or unanswered questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class RagDatasetRecall(BaseModel):
    """Represent the answered questions used as recall ground truth."""

    rag_questions: list[AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources together with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent retrieval results for a complete student dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent retrieval results augmented with generated answers."""

    search_results: list[MinimalAnswer]
    k: int


class AvrageDocumentLengthCache(BaseModel):
    """Store the average token length of indexed chunks."""

    avgdl: float


class ChunksCache(BaseModel):
    """Store the chunks produced by the indexing pipeline."""

    chunks: list[MinimalSource]


class InverseDocumentFrequencyCache(BaseModel):
    """Store inverse document frequency values by token."""

    idf: dict[str, float]


class TermFrequencyCache(BaseModel):
    """Store per-token frequencies keyed by chunk index.."""

    tf: dict[str, dict[str, int]]
