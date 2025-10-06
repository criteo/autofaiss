install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy autofaiss
	python -m pylint autofaiss
	python -m black --check -l 120 autofaiss

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

test: ## [Local development] Run unit tests, doctest and notebooks
	python -m pytest -x -v --cov=autofaiss --cov-report term-missing --cov-fail-under .0001 tests/unit
	# python -m pytest --doctest-modules -v autofaiss

venv-lint-test: ## [Continuous integration]
	python3 -m venv .env && . .env/bin/activate && make install install-dev lint test && rm -rf .env

build-dist: ## [Continuous integration] Build package for pypi
	python3 -m venv .env
	. .env/bin/activate && pip install -U pip setuptools wheel
	. .env/bin/activate && python setup.py sdist
	rm -rf .env

build-pex:
	python3 -m venv .pexing
	. .pexing/bin/activate && python -m pip install -U pip && python -m pip install pex
	. .pexing/bin/activate && python -m pex setuptools "pyspark>=4.0.1,<5.0.0" s3fs>=2022.1.0 "numpy>=1.19.5,<2" . -o autofaiss.pex -v
	rm -rf .pexing

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
