# from transformers import pipeline
from .models import MinimalAnswer, MinimalSearchResults


class Model:
    def __init__(self) -> None:
        name = "Qwen/Qwen3-0.6B"

    def generate(self, search_result: MinimalSearchResults) -> MinimalAnswer:

        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer="nothing yet",
        )
