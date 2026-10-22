import json
from pathlib import Path
from pprint import pprint
import sys
from fire import Fire
from tqdm import tqdm

# from .generation import Model
from .enums import DocumentType, Statics
from .search import Retrieval
from .index import Indexer
from .models import (
    AnsweredQuestion,
    RagDataset,
    Recall,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    MinimalSearchResults,
)


class Rag:

    def index(self, max_chunk_size=2000):
        min_size, overlap_size = (
            Statics.CODE_MIN_CHUNK_SIZE.value,
            Statics.DOCS_OVERLAP_SIZE.value,
        )
        if max_chunk_size <= overlap_size:
            print(f"'max_chunk_size' cannot be less \
                    than overlap size '{overlap_size}'")
            sys.exit()
        elif max_chunk_size <= min_size:
            print(f"'max_chunk_size' cannot be less \
                    than min_chunk_size size '{min_size}'")
            sys.exit()
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

        questions = RagDataset.model_validate_json(
            self.__readfile(dataset_path)
        ).rag_questions

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

        self.__writefile(
            save_directory + "/" + file_path.name,
            results.model_dump_json(indent=4),
        )

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        source_path = Path(student_search_results_path)

        content = StudentSearchResults.model_validate_json(
            self.__readfile(student_search_results_path)
        )

        model = Model()  # type: ignore
        result: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
            search_results=[], k=content.k
        )
        with tqdm(total=len(content.search_results)) as pbar:
            for search_result in content.search_results:
                answer = model.generate(search_result)
                result.search_results.append(answer)
                pbar.update(1)

        self.__writefile(
            save_directory + "/" + source_path.name,
            result.model_dump_json(indent=4),
        )

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        # Recall@k = (Number of queries with target in top-k) / (Total queries)

        results = StudentSearchResults.model_validate_json(
            self.__readfile(student_search_results_path)
        )

        dataset = RagDataset.model_validate_json(self.__readfile(dataset_path))

        ground_truth = {
            answer.question_id: answer.sources  # type: ignore Cannot access attribute "sources" for class "UnansweredQuestion"
            for answer in dataset.rag_questions
        }

        validator = Recall()
        recalls = validator.evaluate(results.search_results, ground_truth)
        result = {f"recall@{k}": v for k, v in recalls.items()}
        return result

    def __readfile(self, file_path: str) -> str:

        try:

            with open(file_path) as f:
                return f.read()

        except OSError as e:
            print("File:", file_path)
            print("Error:", e.strerror)
            sys.exit(1)

        return ""

    def __writefile(self, file_path: str, content: str) -> None:

        try:
            with open(file_path, "w") as f:
                f.write(content)
        except OSError as e:
            print("File:", file_path)
            print("Error:", e.strerror)
            sys.exit(1)

    # def test(self):
    #     from .test import test


if __name__ == "__main__":
    Fire(Rag)
