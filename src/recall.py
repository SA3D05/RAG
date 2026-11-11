import sys

from .models import MinimalSearchResults, MinimalSource
from .utils import Utils


class Recall:
    def __init__(self) -> None:
        self.min_iou = 0.05

    def __get_intersection(
        self, truth_source: MinimalSource, result_source: MinimalSource
    ):

        truth_start = truth_source.first_character_index
        truth_end = truth_source.last_character_index

        result_start = result_source.first_character_index
        result_end = result_source.last_character_index

        return min(result_end, truth_end) - max(result_start, truth_start)

    def __get_union(
        self, truth_source: MinimalSource, result_source: MinimalSource
    ):

        truth_start = truth_source.first_character_index
        truth_end = truth_source.last_character_index

        result_start = result_source.first_character_index
        result_end = result_source.last_character_index

        return max(result_end, truth_end) - min(result_start, truth_start)

    def __check_iou(
        self, truth_source: MinimalSource, result_source: MinimalSource
    ) -> bool:

        intersection = self.__get_intersection(truth_source, result_source)
        union = self.__get_union(truth_source, result_source)

        if union <= 0:
            return False

        if intersection > 0 and intersection / union > self.min_iou:
            return True
        return False

    def __is_found(
        self,
        truth_source: MinimalSource,
        result_source: list[MinimalSource],
    ) -> bool:
        for source in result_source:
            if source.file_path != truth_source.file_path:
                continue
            if self.__check_iou(truth_source, source):
                return True
        return False

    def evaluate(
        self,
        results: list[MinimalSearchResults],
        ground_truth: dict[str, list[MinimalSource]],
    ) -> dict:
        recalls = {
            1: 0.0,
            3: 0.0,
            5: 0.0,
            10: 0.0,
        }
        evaluated_results = 0
        for question_result in results:
            question_id = question_result.question_id
            if question_id not in ground_truth:
                print(f"The question_id {question_id} not fount!")
                sys.exit()

            truth_sources = ground_truth[question_id]

            if not truth_sources:
                Utils.write_err(
                    f"No object found for question_id {question_id}"
                )
                sys.exit()

            if not truth_sources:
                continue
            evaluated_results += 1
            truth_source = truth_sources[0]
            for k in recalls.keys():

                if self.__is_found(
                    truth_source, question_result.retrieved_sources[:k]
                ):
                    recalls[k] = recalls[k] + 1
        if evaluated_results == 0:
            return recalls

        for k, v in recalls.items():
            recalls[k] = v / evaluated_results

        return recalls
