.PHONY: format format-check lint test pipeline

format:
	poetry run black quanttradeai/

format-check:
	poetry run black --check quanttradeai/

lint:
	poetry run flake8 quanttradeai/

test:
	poetry run pytest

pipeline:
	poetry run quanttradeai train -c config/model_config.yaml

