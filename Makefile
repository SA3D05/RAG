TYPE = private
DATA_SETS = data/dataset_$(TYPE)
CHUNK_SIZE = 2000
TOP_K = 10



all: index search grade


index:
	@ uv run python -m src index -max_chunk_size $(CHUNK_SIZE) 
#	2>dev.log

search:
	@ echo "SEARCH_DOCS:"
	@ uv run python -m src \
	search_dataset -dataset_path $(DATA_SETS)/UnansweredQuestions/dataset_docs_$(TYPE).json \
	-k $(TOP_K) \
	-save_directory data/output/search_results/ 

	@ echo "SEARCH_CODE:"
	@ uv run python -m src \
	search_dataset -dataset_path $(DATA_SETS)/UnansweredQuestions/dataset_code_$(TYPE).json \
	-k $(TOP_K) \
	-save_directory data/output/search_results/ 

grade:
	@ echo "-------------------------------------------------"
	@ moulinette/moulinette evaluate_student_search_results \
		data/output/search_results/dataset_docs_$(TYPE).json \
		$(DATA_SETS)/AnsweredQuestions/dataset_docs_$(TYPE).json \
		--k $(TOP_K) 
	@ echo "------------------- DOCS_TEST -------------------"

	@ echo "-------------------------------------------------"
	@ moulinette/moulinette evaluate_student_search_results \
		data/output/search_results/dataset_code_$(TYPE).json \
		$(DATA_SETS)/AnsweredQuestions/dataset_code_$(TYPE).json \
		--k $(TOP_K) 
	@ echo "------------------- CODE_TEST -------------------"
	
test:
	@ uv run python -m src \
	test 2>text.log


clean:
	rm -f data/processed/*
	rm -f data/output/search_results/*
	rm -f data/output/search_results_and_answer/*
