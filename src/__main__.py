from pathlib import Path
import sys
from typing import Any
from fire import Fire
from tqdm import tqdm

# from .generation import Model
from .enums import DocumentType
from .search import Retrieval
from .index import Indexer
from .models import (
    RagDataset,
    RagDatasetRecall,
    Recall,
    Utils,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    MinimalSearchResults,
)


class Rag:

    def index(self, max_chunk_size: Any = 2000):

        max_chunk_size_value = Utils.get_valid_max_chunk_size(max_chunk_size)
        raw_data_path = Path("data/raw")
        indexer = Indexer(raw_data_path, max_chunk_size_value)
        indexer.index()

    def search(self, query: Any, k: Any):
        k_value = Utils.get_valid_k_value(k)

        retrieval = Retrieval()
        result = [
            result.model_dump_json()
            for result in retrieval.search(
                str(query), k_value, DocumentType.DOCS
            )
        ]

        if not result:
            Utils.write_log("No result was found!")

        return result

    def search_dataset(self, dataset_path: Any, k: Any, save_directory: Any):

        k_value = Utils.get_valid_k_value(k)
        output_path = Path(str(save_directory))
        file_path = Path(str(dataset_path))
        retrieval = Retrieval()

        questions = Utils.get_pydantic_model(
            dataset_path, RagDataset
        ).rag_questions

        results: StudentSearchResults = StudentSearchResults(
            search_results=[], k=k_value
        )

        Utils.write_log("retriving ...")
        with tqdm(total=len(questions), file=sys.stdout) as pbar:
            for quest in questions:
                pbar.update(1)

                if not isinstance(quest, UnansweredQuestion):
                    continue

                if "code" in file_path.name:
                    res = retrieval.search(
                        quest.question, k_value, DocumentType.CODE
                    )

                else:
                    res = retrieval.search(
                        quest.question, k_value, DocumentType.DOCS
                    )

                results.search_results.append(
                    MinimalSearchResults(
                        question_id=quest.question_id,
                        question=quest.question,
                        retrieved_sources=res,
                    )
                )

        Utils.writefile(
            str(output_path) + "/" + file_path.name,
            results.model_dump_json(indent=4),
        )

    def answer_dataset(
        self, student_search_results_path: Any, save_directory: Any
    ):
        source_path = Path(student_search_results_path)

        content = Utils.get_pydantic_model(
            student_search_results_path, StudentSearchResults
        )

        model = Model()  # type: ignore
        result: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
            search_results=[], k=content.k
        )
        with tqdm(total=len(content.search_results), file=sys.stdout) as pbar:
            for search_result in content.search_results:
                answer = model.generate(search_result)
                result.search_results.append(answer)
                pbar.update(1)

        Utils.writefile(
            save_directory + "/" + source_path.name,
            result.model_dump_json(indent=4),
        )

    def evaluate(self, student_search_results_path: Any, dataset_path: Any):
        results = Utils.get_pydantic_model(
            student_search_results_path, StudentSearchResults
        )
        dataset = Utils.get_pydantic_model(dataset_path, RagDatasetRecall)

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
        Utils.write_err("Exit program")
