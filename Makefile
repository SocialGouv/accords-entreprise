test:
	poetry run pytest tests/unit
	python scripts/check_coverage.py coverage.json
	@echo "===== Finished running unit tests ====="

integration_test:
	poetry run pytest tests/integration
	@echo "===== Finished running integration tests ====="

install:
	if command -v apt-get >/dev/null; then \
		sudo apt-get update && sudo apt-get install -y antiword; \
	elif command -v yum >/dev/null; then \
		sudo yum install -y antiword; \
	elif command -v brew >/dev/null; then \
		brew install antiword; \
	else \
		echo "Package manager not found. Please install antiword manually."; \
		exit 1; \
	fi
	poetry install
	poetry run pre-commit install --allow-missing-config -f
	poetry run detect-secrets scan > .secrets.baseline
	alembic upgrade head

run-ingest:
	@set -x
	python ./scripts/ingest_documents.py
	@set +x

run-theming:
	@set -x
	python ./scripts/theme_documents.py
	@set +x
