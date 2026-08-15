# Test suite

Run the normal development suite from the repository root:

```bash
pytest
```

The default run is self-contained. Tests needing a live service, an external
dataset, or an external executable such as ORCA are marked `integration` and
require an explicit opt-in:

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
