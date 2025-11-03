.PHONY: install lint format test train

install:
	python -m pip install -U pip && pip install -e .[dev]

lint:
	ruff check src tests

format:
	black src tests

test:
	pytest -q

train:
	hwr-train --epochs 3 --batch-size 128 --lr 1e-3