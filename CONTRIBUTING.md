# Contributing to AssemblyTheoryTools

Thank you for considering a contribution to AssemblyTheoryTools. Bug reports,
documentation improvements, tests, and code changes are all welcome.

## Getting started

### Reporting bugs and requesting features

For substantial bug fixes or new features, open an
[issue](https://github.com/ELIFE-ASU/assemblytheorytools/issues) before starting
a pull request.

- For bugs, include a minimal reproducible example, your operating system, the
  ATT version, and your Python version.
- For features, explain the motivation and how the proposal fits the existing
  architecture. Wait for feedback before investing in a large change.

### Development environment

The project supports Python 3.12 and newer. Fork and clone the repository,
create an isolated environment, then install the package and development tools:

```console
python -m pip install -e ".[dev]"
```

## Development workflow

1. Create a focused branch:

   ```console
   git switch -c feature/my-new-feature
   ```

2. Implement the change and add or update tests and documentation.
3. Run the relevant checks locally.
4. Commit with a concise message that explains the change.
5. Push the branch to your fork and open a pull request against `main`.

Keep pull requests focused. Separate unrelated refactoring from functional
changes so each review remains understandable.

## Coding guidelines

- Use features supported by Python 3.12 and newer.
- Package reusable behaviour in functions or classes and avoid unnecessary
  global state.
- Add type hints to public function signatures and class attributes.
- Write NumPy-style docstrings for every public class and function, including
  parameters, return values, and raised exceptions.
- Keep docstring lines at or below 76 characters so API documentation renders
  cleanly.

## Testing

Tests use the third-party [pytest](https://docs.pytest.org/) framework and live
under `tests/`. Add focused regression tests alongside every bug fix or new
behaviour.

Run the default, self-contained test suite:

```console
pytest
```

Integration tests need a live service, external data, or an executable such as
ORCA, and slow tests are excluded from the normal development loop. Enable them
explicitly when the change requires them:

```console
pytest --run-integration
pytest --run-slow
pytest --run-integration --run-slow
```

Generate branch coverage for the default suite:

```console
pytest --cov --cov-report=term-missing
```

New work should not reduce existing coverage. See the
[test-suite guide](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/tests/README.md)
for marker and plotting details.

## Documentation

The documentation lives in `docs/` and is published at
[assemblytheorytools.readthedocs.io](https://assemblytheorytools.readthedocs.io/).
Install the documentation extra and run the strict build:

```console
python -m pip install -e ".[docs]"
make -C docs strict
```

The strict target treats warnings as errors and writes the HTML output to
`docs/build/html`. Autodoc imports the real package, so missing runtime
dependencies and broken docstrings fail the build.

When changing documentation:

- Write narrative pages in MyST Markdown; keep API stubs under
  `docs/source/api/` in reStructuredText.
- Add every new page to a `toctree`.
- Check that user-guide snippets run as written.
- Add every new public function or class to its module's `autosummary` in
  `docs/source/api/*.rst`.
- The protocol pages are the executed notebooks under `examples/protocols/`,
  rendered by MyST-NB from their committed outputs; the build never runs them.
  After editing one, re-execute it with
  `jupyter nbconvert --to notebook --execute --inplace protocol_N.ipynb` from
  its own directory so the stored outputs stay current.

## Pull request checklist

- [ ] The change is focused and excludes unrelated refactoring.
- [ ] New behaviour and bug fixes have tests.
- [ ] The relevant default, integration, or slow test groups pass.
- [ ] Coverage does not decrease.
- [ ] Public APIs have type hints and NumPy-style docstrings.
- [ ] User-facing behaviour is documented.
- [ ] `make -C docs strict` passes when documentation or public APIs change.
- [ ] New public objects appear in the appropriate API `autosummary`.
