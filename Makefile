
MAX_LENGTH=2000
ACTION=index
K=5

run: index search grade test_grade

test_grade: grade grademe

index: install
	@uv run python3 -m src index \
	-max_chunk_size $(MAX_LENGTH)

search: install
	@echo "=========================== {SEARCH DOCS} ==========================="

	@uv run python3 -m src search_dataset \
	-dataset_path data/datasets/private/UnansweredQuestions/dataset_docs_private.json \
	-k $(K) \
	-save_directory data/output/

	@echo "=========================== {SEARCH CODE} ==========================="

	@uv run python3 -m src search_dataset \
	-dataset_path data/datasets/private/UnansweredQuestions/dataset_code_private.json \
	-k $(K) \
	-save_directory data/output/

grade:
	@./moulinette evaluate_student_search_results \
	data/output/dataset_docs_private.json \
	data/datasets/private/AnsweredQuestions/dataset_docs_private.json \
	--k $(K) \
	--max_context_length $(MAX_LENGTH)
	
	@echo "=========================== {GRADE DOCS} ==========================="

	@./moulinette evaluate_student_search_results \
	data/output/dataset_code_private.json \
	data/datasets/private/AnsweredQuestions/dataset_code_private.json \
	--k $(K) \
	--max_context_length $(MAX_LENGTH)

	@echo "=========================== {GRADE CODE} ==========================="

grademe:
	@uv run python3 -m src evaluate \
	data/output/dataset_docs_private.json \
	data/datasets/private/AnsweredQuestions/dataset_docs_private.json
	@echo "=========================== {GRADEME DOCS} ==========================="

	@uv run python3 -m src evaluate \
	data/output/dataset_code_private.json \
	data/datasets/private/AnsweredQuestions/dataset_code_private.json
	@echo "=========================== {GRADEME CODE} ==========================="

install:
	uv sync

debug: install
	uv run python3 -m pdb -m src $(ACTION)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache \
	data/output/* \
	data/processed/*
	

lint: install
	uv run mypy -m src --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs
	uv run flake8 src


