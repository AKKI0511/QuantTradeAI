.PHONY: format format-check lint test pipeline

format:
	poetry run black quanttradeai/

format-check:
	poetry run black --check quanttradeai/

lint:
	poetry run flake8 --ignore=E501 quanttradeai/

test:
	poetry run python -c "import os; os.environ.setdefault('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1'); import pytest; raise SystemExit(pytest.main(['-p', 'pytest_asyncio.plugin']))"

pipeline:
	poetry run quanttradeai train -c config/model_config.yaml

