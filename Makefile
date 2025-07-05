.PHONY: format lint test pipeline

format:
	poetry run black .

lint:
	poetry run flake8

test:
	poetry run pytest

pipeline:
	poetry run python -m src.main train -c config/model_config.yaml
