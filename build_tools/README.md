The Conda environments provide Python, Git and a C++ compiler for ATT's
on-demand parallelassemblycpp build, plus Cairo for rendering. Python
dependencies, including CMake and Ninja, come from the package metadata in
`pyproject.toml`. Both environments use only conda-forge and select Python
3.12–3.14, the versions tested in CI.

Run the commands below from the root of this checkout.

For the published package, run:

```sh
conda env create -f build_tools/environment.yml
conda activate att_env
```

For an editable development installation:

```sh
conda env create -f build_tools/environment_dev.yml
conda activate att_dev_env
python -m pip install --group build --group lint
```

The development environment installs the package with its `dev`, `docs` and
`notebooks` extras, so changes to the shared dependency declarations apply here
as well as in CI.

The `build` and `lint` dependency groups provide the package build, distribution
validation and lint tools. Installing groups requires pip 25.1 or newer, which
the development environment supplies. Build and check a release locally with:

```sh
python -m build
python -m twine check --strict dist/*
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the test and documentation checks.
