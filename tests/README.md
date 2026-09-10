# Test suite

Run the normal development suite from the repository root:

```bash
pytest
```

The default run needs no live service and no external dataset, but it does need
the C++ calculator. That is not shipped as a binary, so the first test that uses
it builds parallelassemblycpp from source, which takes a few minutes and needs `git`
and a C++20 compiler. Point `ASS_PATH` at an executable you already have to skip
that:

```bash
ASS_PATH=/path/to/AssemblyCpp pytest
```

The on-demand build lands in `.pytest_cache/runtime/assemblytheorytools/`,
because `conftest.py` redirects `XDG_CACHE_HOME` there; delete that directory to
force a rebuild.

Tests needing a live service, an external dataset, or an external executable
such as ORCA are marked `integration` and require an explicit opt-in:

```bash
pytest --run-integration
```

Long-running local calculations are marked `slow`:

```bash
pytest --run-slow
```

Both groups can be enabled together. The marker expression remains useful for
selecting a subset, while the flags grant permission for that group to run:

```bash
pytest --run-integration --run-slow -m "integration or slow"
```

Generate branch coverage for the normal suite with:

```bash
pytest --cov --cov-report=term-missing
```

Plotting entry points are stubbed by default so the suite remains headless. Set
`ATT_TEST_SHOW_PLOTS=1` when manually inspecting figures and images.

## Organization

Keep tests beside the behavior they exercise, including regressions. Use
`test_<module>.py` for most modules. Larger areas have a few focused suites:

| Area | Suites |
| --- | --- |
| Assembly calculations | `test_assembly_mols.py`, `test_assembly_strings.py` |
| Assembly backends | `test_assembly_backends.py`, `test_assembly_rust.py` |
| Ensemble quantities | `test_assembly_ensemble.py` |
| Data utilities | `test_tools_data.py`, `test_tools_data_pubchem.py`, `test_tools_data_spectra.py` |

Avoid separate `*_refactor` or `*_regressions` files. A regression's name or a
short comment should explain the behavior it protects.

## Writing tests

- Give each test a descriptive behavior name. Parameterize independent examples
  of the same contract, with readable case IDs when the values are complex.
- Assert observable results, including order and multiplicity when meaningful.
  Use `pytest.approx` or NumPy assertions for floating-point values. Bounds must
  exclude failure sentinels such as `-1`.
- Keep fixtures small and local. Share them in `conftest.py` only when multiple
  modules need them. Use `data_dir` for bundled inputs and `tmp_path` for outputs;
  use `monkeypatch` for environment changes, working directories and stubs.
- Test timeout handling with controlled clocks or backend responses. Retain real
  backend smoke checks, but avoid assertions that depend on machine speed.
- Keep plotting assertions in plotting tests, using deterministic input and
  checking artists or saved artifacts. Avoid debug prints and unasserted plots.

`conftest.py` seeds Python and NumPy's global generators to zero before each test
and restores their previous states afterward. Set a different seed explicitly
when it defines a regression case. Display stubs are also restored after each
test, and figures are closed automatically in headless runs.

`serial_data_mp` keeps data transformations in the test process; the dedicated
multiprocessing suite verifies real process and thread pools. Integration tests
that need ORCA use `orca_path`, resolved from `ORCA_PATH` or the executable search
path, and skip when it is unavailable.

`orca_path` also runs the candidate once per session and requires it to print an
ORCA version banner before handing it to a test. Finding a program called `orca`
is not enough: Ubuntu's `orca` package is GNOME Orca, the accessibility screen
reader, installed at `/usr/bin/orca` and unrelated to the quantum-chemistry
ORCA. Without the check it reached ASE, which spent minutes on it and failed
with a bare non-zero exit status — a misconfigured environment that reads like a
flaky test. Prefer setting `ORCA_PATH` explicitly:

```bash
ORCA_PATH=$HOME/orca_6_1_1/orca pytest --run-integration
```
