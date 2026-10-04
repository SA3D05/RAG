

INPUT_FILE = dataset_docs_public.json
OUTPUT_FILE = dataset_public.json
CHUNK_SIZE = 2000
TOP_K = 10



all: index search grade


index:
	@ uv run python -m src index -max_chunk_size $(CHUNK_SIZE) 
#	2>dev.log

search:
	@ uv run python -m src \
	search_dataset -dataset_path data/datasets/UnansweredQuestions/$(INPUT_FILE) \
	-k $(TOP_K) \
	-save_directory data/output/search_results/ 

grade:
	@ moulinette/moulinette evaluate_student_search_results \
		data/output/search_results/$(OUTPUT_FILE) \
		data/datasets/AnsweredQuestions/$(INPUT_FILE) \
		--k $(TOP_K) 
test:
	@ uv run python3 src/test.py 


clean:
	rm -f data/processed/*
	rm -f data/output/search_results/*
	rm -f data/output/search_results_and_answer/*
