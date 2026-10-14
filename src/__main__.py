from pathlib import Path
from fire import Fire
from tqdm import tqdm
from .generation import Model
from .enums import DocumentType
from .search import Retrieval
from .index import Indexer
from .models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    MinimalSearchResults,
)


class Rag:

    def index(self, max_chunk_size=2000):
        Indexer.index(max_chunk_size)

    def search(self, query: str, k: int):
        retrieval = Retrieval()
        return [
            result.model_dump_json()
            for result in retrieval.search(query, k, DocumentType.DOCS)
        ]

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        file_path = Path(dataset_path)  # $(DATA_SETS)/UnansweredQuestions/
        retrieval = Retrieval()

        with open(file_path) as f:
            questions = RagDataset.model_validate_json(f.read()).rag_questions

        results: StudentSearchResults = StudentSearchResults(
            search_results=[], k=k
        )
        print("retriving ...")
        with tqdm(total=len(questions)) as pbar:
            for quest in questions:
                pbar.update(1)

                if not isinstance(quest, UnansweredQuestion):
                    continue

                if "docs" in file_path.name:
                    res = retrieval.search(
                        quest.question, k, DocumentType.DOCS
                    )

                else:
                    res = retrieval.search(
                        quest.question, k, DocumentType.CODE
                    )

                results.search_results.append(
                    MinimalSearchResults(
                        question_id=quest.question_id,
                        question=quest.question,
                        retrieved_sources=res,
                    )
                )

        with open(
            save_directory + "/" + file_path.name,
            "w",
        ) as f:
            f.write(results.model_dump_json(indent=4))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        source_path = Path(student_search_results_path)
        with open(source_path) as f:
            content = StudentSearchResults.model_validate_json(f.read())

        model = Model()
        result: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
            search_results=[], k=content.k
        )
        with tqdm(total=len(content.search_results)) as pbar:
            for search_result in content.search_results:
                answer = model.generate(search_result)
                result.search_results.append(answer)
                pbar.update(1)

        with open(save_directory + "/" + source_path.name, "w") as f:
            f.write(result.model_dump_json(indent=4))

    def test(self):
        from .test import test


if __name__ == "__main__":
    Fire(Rag)
