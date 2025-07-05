.PHONY: format format-check lint test pipeline

format:
	poetry run black src/

format-check:
	poetry run black --check src/

lint:
	poetry run flake8 src/

test:
	poetry run pytest

pipeline:
	poetry run python -m src.main train -c config/model_config.yaml
