

all: index search grade


index:
	uv run python -m src index -max_chunk_size 2000 2>dev.log

search:
	uv run python -m src \
	search_dataset -dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
	-k 10 \
	-save_directory data/output/search_results/ 2>dev.log

grade:
	moulinette/moulinette evaluate_student_search_results \
		data/output/search_results/dataset_docs_public.json \
		data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10 2>dev.log
test:
	uv run python3 src/test.py 