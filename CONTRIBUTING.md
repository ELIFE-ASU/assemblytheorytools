# Contributing to `AssemblyTheoryTools`

For larger bug fixes or new features, please file an issue before submitting a
pull request. If the change isn't trivial, it may be best to wait for
feedback.

Please follow standard Python practices, write clear commit messages, and ensure all code is well-documented and tested.
To contribute, branch the repo, make changes, and submit a pull request.

Contribution checklist:

- Code must be packaged into reusable functions or classes.
- Code must use current tooling.
- Code must have documentation.
- Code must have at least one passing test.

## Running tests

Tests are written as usual Python unit tests with the `unittest` module of
the standard library. They can be run the usual way:

```console
$ python -m unittest discover -vv
```

## Coding guidelines

This project targets Python 3.8 or later.

### Docstrings

The docstring lines should not be longer than 76 characters (which allows rendering without soft-wrap of the entire
module in a 80x24 terminal window).  
Docstrings should be written in Google format.
