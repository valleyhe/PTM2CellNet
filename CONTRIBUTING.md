# Contributing to PTM2CellNet

Thank you for your interest in contributing to PTM2CellNet! This guide covers
how to report bugs, request features, set up a development environment, and
submit pull requests.

## Bug Reports

If you find a bug, please open a [GitHub Issue](../../issues) and include:

- **Python version** and OS
- **Package versions** (`pip freeze` output or relevant subset)
- **Minimal reproducible example** — the smallest code snippet that triggers the issue
- **Expected vs. actual behavior**
- **Full traceback** if an exception is raised

## Feature Requests

Open a [GitHub Issue](../../issues) with the label `enhancement`. Describe:

- The problem or limitation you want to address
- The proposed solution or API
- Any alternatives you have considered

## Development Setup

1. **Fork and clone** the repository.

2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-core.txt
   pip install -r requirements-pretrained.txt   # optional, for PLM features
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (if configured):

   ```bash
   pre-commit install
   ```

## Code Style

- Follow **PEP 8** as configured by Ruff, with line length up to 120 characters (`pyproject.toml`).
- Use **type annotations** for all public functions and methods.
- Run the project checks before submitting:

  ```bash
  ruff check src scripts tests
  python -m ruff format --check src scripts tests
  python -m mypy src/ --ignore-missing-imports
  python scripts/check_requirements_consistency.py
  ```

## Testing

Run the default offline regression before opening a pull request:

```bash
python -m pytest -m "not slow and not gpu" --timeout=300
```

For a quicker check during development:

```bash
python -m pytest -x --tb=short -m "not slow and not gpu" --timeout=300
```

## Pull Request Process

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits.

3. **Add or update tests** that cover your changes.

4. **Run the project checks**:

   ```bash
   python -m pytest -m "not slow and not gpu" --timeout=300
   ruff check src scripts tests
   python -m ruff format --check src scripts tests
   python -m mypy src/ --ignore-missing-imports
   python scripts/check_requirements_consistency.py
   ```

5. **Update documentation** if your change affects the public API or user-facing behavior.

6. **Open a pull request** against `main` with:
   - A clear description of the change
   - Reference to any related issues (`Fixes #123` or `Closes #456`)
   - Confirmation that tests pass

7. **Address review feedback** promptly. A maintainer will merge once approved.

## License

By contributing to PTM2CellNet, you agree that your contributions will be
licensed under the [MIT License](LICENSE).
