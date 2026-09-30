

all: run

run:
	@ systemd-run --user --scope \
	-p MemoryMax=1G \
	-p CPUQuota=100% \
	uv run python3 -m src

moulinette:
	./moutlinette/moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 2 \
		--max_context_length 2000