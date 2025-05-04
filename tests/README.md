# Tests for Doctor Application

This directory contains tests for the Doctor application.

## Structure

- `conftest.py`: Common fixtures for all tests
- `lib/`: Tests for the library components
  - `test_crawler.py`: Tests for the crawler module
  - `test_chunker.py`: Tests for the chunker module
  - `test_embedder.py`: Tests for the embedder module
  - `test_indexer.py`: Tests for the indexer module
  - `test_database.py`: Tests for the database module
  - `test_processor.py`: Tests for the processor module

## Running Tests

To run all tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src
```

To run specific test categories:

```bash
# Run all unit tests
pytest -m unit

# Run all async tests
pytest -m async_test

# Run tests for a specific module
pytest tests/lib/test_crawler.py
```

## Test Markers

- `unit`: Unit tests
- `integration`: Integration tests (not implemented yet)
- `async_test`: Tests that use asyncio 