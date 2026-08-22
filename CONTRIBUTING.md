# Contributing to `AssemblyTheoryTools`

First off, thank you for considering contributing to `AssemblyTheoryTools`\! It's people like you that make this tool
better for everyone.

We welcome contributions from everyone. By participating in this project, you agree to abide by our Code of Conduct (if
applicable).

## 1\. Getting Started

### Reporting Bugs & Requesting Features

For larger bug fixes or new features, **please file an issue before submitting a pull request.**

* **Bugs:** Include a minimal reproducible example, your OS, and your Python version.
* **Features:** Explain the motivation and how it fits into the existing architecture. If the change isn't trivial, it
  is best to wait for feedback on the issue to avoid wasted effort.

### Setting Up Your Development Environment

This project targets **Python 3.12+**. Please ensure your local environment matches this requirement.

1. **Fork and Clone** the repository to your local machine.
2. **Create an Environment** to keep dependencies isolated.
3. **Install Dependencies**.
   ```console
   $ pip install -e .[dev]
   ```

-----

## 2\. Development Workflow

1. **Create a Branch:** Always create a new branch for your changes.
   ```console
   $ git checkout -b feature/my-new-feature
   ```
2. **Make Changes:** Implement your feature or bug fix.
3. **Run Tests:** Ensure all tests pass locally before pushing (see "Running Tests" below).
4. **Commit:** Write clear, concise commit messages.
    * *Bad:* "Fixed stuff"
    * *Good:* "Fix(parser): Handle empty strings in input validation"
5. **Push:** Push your branch to your fork.
6. **Pull Request:** Open a PR against the `main` branch of the original repository.

-----

## 3\. Coding Guidelines

We enforce strict standards to maintain code quality and readability.

### General Standards

* **Modern Python:** Use features available in Python 3.12+ (e.g., modern type hinting syntax).
* **Modularity:** New functionality must be packaged into reusable functions or classes. Avoid global state.
* **Type Hints:** All function signatures and class attributes **must** have type hints.

### Docstrings (Numpy Style)

We use the **Numpy** docstring format.

* **Line Length:** Docstring lines must not exceed **76 characters** to ensure clean rendering in standard terminals.
* **Content:** Every public class and function must have a docstring explaining parameters, returns, and errors raised.

**Example:**

```python
def calculate_assembly_index(molecule_data: dict, threshold: float = 0.5) -> int:
    """
    Calculates the assembly index based on molecular graph data.

    Parameters
    ----------
    molecule_data : dict
        A dictionary containing graph edges and nodes.
    threshold : float, optional
        The cutoff value for filtering noise (default is 0.5).

    Returns
    -------
    int
        The calculated assembly index value.
    """
    pass
```

-----

## 4\. Testing

We take testing seriously. Code changes must not reduce existing test coverage.

### Writing Tests

* Tests are written using the standard library `pytest` module.
* Place new tests in the `tests/` directory, mirroring the structure of the source code.

### Running Tests

You can run the full suite using pytest.

### Checking Coverage

We recommend checking coverage locally to ensure you haven't unknowingly altered other code.

-----

## 5\. Documentation

The documentation lives in `docs/` and is published at
[assemblytheorytools.readthedocs.io](https://assemblytheorytools.readthedocs.io/).

### Building it locally

```console
$ pip install -e ".[docs]"
$ make -C docs strict
```

The `strict` target treats warnings as errors, which is exactly what Read the Docs runs. A clean local build therefore
means a clean remote one. The output lands in `docs/build/html`.

Nothing is mocked: autodoc imports the real package, so a missing runtime dependency or a broken docstring surfaces as a
build failure rather than a silently empty page.

### Adding a page

* Narrative pages are MyST Markdown. The API stubs under `docs/source/api/` stay reStructuredText.
* Every new page needs an entry in a `toctree`, or the build fails on an orphaned document.
* Code snippets in the user guide are expected to run as written. Check yours against a real environment before
  submitting.
* When you add a public function, add it to the `autosummary` list at the top of its module's `docs/source/api/*.rst`
  page.

-----

## 6\. Pull Request Checklist

Before submitting your PR, please ensure you have checked the following boxes:

- [ ] **Scope:** The changes are focused (do not include unrelated refactoring).
- [ ] **Tests:** I have added unit tests for new functionality.
- [ ] **Coverage:** I have verified that my changes do not decrease test coverage.
- [ ] **Documentation:** I have added/updated docstrings (Numpy style, max 76 chars width).
- [ ] **Docs build:** `make -C docs strict` passes, and any new public function appears in its module's `autosummary`.
- [ ] **Types:** I have added type hints to all new functions and classes.
- [ ] **Tooling:** I have used current project tooling and dependencies.

