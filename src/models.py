from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    tokens_len: int = 0


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
