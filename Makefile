TYPE = public
DATA_SETS = data/datasets
CHUNK_SIZE = 2000
TOP_K = 10



all: index search grade answer


index:
	@ uv run python -m src index -max_chunk_size $(CHUNK_SIZE) 
#	2>dev.log

search:
	@ echo "SEARCH_DOCS:"
	@ uv run python -m src \
	search_dataset -dataset_path $(DATA_SETS)/UnansweredQuestions/dataset_docs_$(TYPE).json \
	-k $(TOP_K) \
	-save_directory data/output/search_results/UnansweredQuestions

	@ echo "SEARCH_CODE:"
	@ uv run python -m src \
	search_dataset -dataset_path $(DATA_SETS)/UnansweredQuestions/dataset_code_$(TYPE).json \
	-k $(TOP_K) \
	-save_directory data/output/search_results/UnansweredQuestions

grade:
	@ echo "-------------------------------------------------"
	@ ./moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_docs_$(TYPE).json \
		$(DATA_SETS)/AnsweredQuestions/dataset_docs_$(TYPE).json \
		--k $(TOP_K) 
	@ echo "------------------- DOCS_TEST -------------------"

	@ echo "-------------------------------------------------"
	@ ./moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_code_$(TYPE).json \
		$(DATA_SETS)/AnsweredQuestions/dataset_code_$(TYPE).json \
		--k $(TOP_K) 
	@ echo "------------------- CODE_TEST -------------------"



answer:
	@uv run python -m src answer_dataset \
	--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_$(TYPE).json \
	--save_directory data/output/search_results_and_answer/UnansweredQuestions

test:
	@ uv run python -m src \
	test 2>text.log


clean:
	rm -f data/processed/*
	rm -f data/output/search_results/UnansweredQuestions/*
	rm -f data/output/search_results_and_answer/UnansweredQuestions/*
