# Changelog

Releases are published on GitHub, and each carries its own notes:

**[github.com/ELIFE-ASU/assemblytheorytools/releases](https://github.com/ELIFE-ASU/assemblytheorytools/releases)**

Every release is also pushed to
[PyPI](https://pypi.org/project/assemblytheorytools/), so upgrading is:

```bash
pip install --upgrade assemblytheorytools
```

## Unreleased

* Update the C++ bridge for current parallelassemblycpp, including
  `AssemblyCppOptions(algorithm="re-pair")` for graph and string heuristic
  upper bounds and the graph-only `upper_bound="graph-repair"` compatibility
  selector. Re-Pair certificates are returned as ATT construction pathways.
* Support Unicode code points, native parallel search, explicit thread counts
  and verbose logging in C++ string mode. Omit the graph-only hydrogen flag
  when submitting strings.
* Cover every current CLI flag in the configuration reference and validate
  incompatible input modes, Re-Pair options and parallel search limits before
  launching the calculator.

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
