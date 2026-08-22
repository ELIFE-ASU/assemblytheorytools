# Changelog

Releases are published on GitHub, and each carries its own notes:

**[github.com/ELIFE-ASU/assemblytheorytools/releases](https://github.com/ELIFE-ASU/assemblytheorytools/releases)**

Every release is also pushed to
[PyPI](https://pypi.org/project/assemblytheorytools/), so upgrading is:

```bash
pip install --upgrade assemblytheorytools
```

## Versioning

The version is single-sourced from `pyproject.toml` and exposed at runtime:

```python
import assemblytheorytools as att

print(att.__version__)
```

The documentation you are reading is built from the installed package, so the
version in the sidebar matches the API described here.

## Reporting a problem with a release

Open an issue on the
[tracker](https://github.com/ELIFE-ASU/assemblytheorytools/issues) with the
output of `att.__version__`, your Python version and OS, and a minimal
reproduction. See {doc}`contributing`.
