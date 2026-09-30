import json
from pathlib import Path
from pprint import pprint
import sys
from .search import Retrieval
from .index import Indexer
from .models import (
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
    MinimalSearchResults,
)

datasets_path = Path("data/datasets_public/public")

datasets: list[RagDataset] = []
for unanswered in datasets_path.rglob("UnansweredQuestions/*.json"):
    with open(unanswered) as f:
        datasets.append(RagDataset.model_validate_json(f.read()))

# for dataset in datasets:
questions = datasets[-1].rag_questions

k_value = 2
results: StudentSearchResults = StudentSearchResults(
    search_results=[], k=k_value
)

for quest in questions:

    if not isinstance(quest, UnansweredQuestion):
        continue

    res = Retrieval.search(quest.question, k_value)
    # with open("test_result.json", "a") as f:
    #     json.dump([source.model_dump() for source in res], f, indent=4)
    results.search_results.append(
        MinimalSearchResults(
            question_id=quest.question_id,
            question=quest.question,
            retrieved_sources=res,
        )
    )

with open(
    "data/output/search_results/UnansweredQuestions/dataset_docs_public.json",
    "w",
) as f:
    f.write(results.model_dump_json(indent=4))

# for answeredin datasets_path.rglob("AnsweredQuestions/*.json" ):
#     pass

# query = "openai"
# Indexer.index()
