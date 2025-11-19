# Contributing to `AssemblyTheoryTools`

For larger bug fixes or new features, please file an issue before submitting a
pull request. If the change isn't trivial, it may be best to wait for
feedback.

Please follow standard Python practices, write clear commit messages, and ensure all code is well-documented and tested.
To contribute, branch the repo, make changes, and submit a pull request.

Contribution checklist:

- New functionality must be packaged into reusable functions or classes.
- New functionality must use current tooling.
- New functionality must have type hints.
- New functionality must have docstrings.
- New functionality must have unit tests.
- Code changes must not reduce existing test coverage.

## Running tests

Tests are written as usual Python unit tests with the `unittest` module of
the standard library. They can be run the usual way:

```console
$ python -m unittest discover -vv
```

## Coding guidelines

This project targets Python 3.12 or later, please ensure your code is compatible.

### Docstrings

The docstring lines should not be longer than 76 characters (which allows rendering without soft-wrap of the entire
module in a 80x24 terminal window).  
Docstrings should be written in Numpy style format.
