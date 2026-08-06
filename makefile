run:
	...

install:
	...

index:
	uv run python3 -m src index

search:
	uv run python3 -m src

lint:
	flake8 .
	mypy . \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

clean:
	@rm -rf data/output/search_results
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf __pycache__
	@rm -rf src/__pycache__
	@rm -rf .nvim

