from pathlib import Path
import sys
from fire import Fire
from pydantic import ValidationError
from tqdm import tqdm

# from .generation import Model
from .enums import DocumentType, Statics
from .search import Retrieval
from .index import Indexer
from .models import (
    RagDataset,
    RagDatasetRecall,
    Recall,
    FileOP,
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

        try:

            if max_chunk_size > 2000:
                raise ValueError(f"'max_chunk_size' grater than 2000")
            if max_chunk_size <= overlap_size:
                raise ValueError(
                    f"'max_chunk_size' cannot be less than overlap size {overlap_size}"
                )
            elif max_chunk_size <= min_size:
                raise ValueError(
                    f"'max_chunk_size' cannot be less than min_chunk_size size {min_size}"
                )

        except ValueError as e:
            print("Error:", e)
            sys.exit(1)

        Indexer.index(max_chunk_size)

    def search(self, query: str, k: int):
        try:
            int(k)
        except ValueError:
            print(
                "Error:",
                f"Invalid 'k' value '{k}'",
            )
            sys.exit(1)

        retrieval = Retrieval()
        result = [
            result.model_dump_json()
            for result in retrieval.search(str(query), k, DocumentType.DOCS)
        ]
        if not result:
            print("No result was found!")
        return result

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        try:
            int(k)
        except ValueError:
            print(
                "Error:",
                f"Invalid 'k' value '{k}'",
            )
            sys.exit(1)

        output_path = Path(str(save_directory))
        file_path = Path(str(dataset_path))
        retrieval = Retrieval()

        questions = FileOP.get_pydantic_model(
            dataset_path, RagDataset
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

        FileOP.writefile(
            str(output_path) + "/" + file_path.name,
            results.model_dump_json(indent=4),
        )

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        source_path = Path(student_search_results_path)

        content = FileOP.get_pydantic_model(
            student_search_results_path, StudentSearchResults
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

        FileOP.writefile(
            save_directory + "/" + source_path.name,
            result.model_dump_json(indent=4),
        )

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        # Recall@k = (Number of queries with target in top-k) / (Total queries)

        results = FileOP.get_pydantic_model(
            student_search_results_path, StudentSearchResults
        )

        dataset = FileOP.get_pydantic_model(dataset_path, RagDatasetRecall)

        ground_truth = {
            answer.question_id: answer.sources
            for answer in dataset.rag_questions
        }

        validator = Recall()
        recalls = validator.evaluate(results.search_results, ground_truth)
        result = {f"recall@{k}": v for k, v in recalls.items()}
        return result


if __name__ == "__main__":
    try:
        Fire(Rag)
    except KeyboardInterrupt:
        print("Exit program")
