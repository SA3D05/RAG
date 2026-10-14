

ACTION=index


run: install
	uv run python3 -m src $(ACTION)

install:
	uv sync

debug: install
	uv run python3 -m pdb -m src $(ACTION)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache \
	data/output/search_results/UnansweredQuestions/* \
	data/processed/*
	

lint: install
	uv run mypy -m src --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs
	uv run flake8 src