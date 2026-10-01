from fire import Fire
from .search import Retrieval
from .index import Indexer
from .models import (
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
    MinimalSearchResults,
)


class Rag:

    def index(self, max_chunk_size=2000):
        Indexer.index(max_chunk_size)

    def search(self, query: str, k: int):
        return [
            result.model_dump_json() for result in Retrieval.search(query, k)
        ]

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):

        print(f"{dataset_path!r}")
        with open(dataset_path) as f:
            questions = RagDataset.model_validate_json(f.read()).rag_questions
        results: StudentSearchResults = StudentSearchResults(
            search_results=[], k=k
        )

        for quest in questions:

            if not isinstance(quest, UnansweredQuestion):
                continue

            res = Retrieval.search(quest.question, k)
            results.search_results.append(
                MinimalSearchResults(
                    question_id=quest.question_id,
                    question=quest.question,
                    retrieved_sources=res,
                )
            )

        with open(
            save_directory + "/dataset_docs_public.json",
            "w",
        ) as f:
            f.write(results.model_dump_json(indent=4))


if __name__ == "__main__":
    Fire(Rag)
