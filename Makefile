.PHONY: dev lint format

UV := uv run

# Install/sync dev dependencies (run once or after dependency changes).
dev:
	uv sync --group dev

# Apply Black and isort (writes files). Use alone when you want formatting without mypy.
format:
	$(UV) black .
	$(UV) isort .

# Apply Black/isort, then run mypy on `aieng`. Auto-fixes what Black/isort can; mypy only reports.
lint: format
	$(UV) mypy -p aieng
