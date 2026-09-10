The Conda environments provide Python, Git and a C++ compiler for ATT's
on-demand parallelassemblycpp build, plus Cairo for rendering. Python
dependencies, including CMake and Ninja, come from the package metadata in
`pyproject.toml`.

For the published package, run:

```sh
conda env create -f build_tools/environment.yml
conda activate att_env
```

For an editable development installation, run these commands from the root of
this checkout:

```sh
conda env create -f build_tools/environment_dev.yml
conda activate att_dev_env
```

The development environment installs the package with its `dev`, `docs` and
`notebooks` extras, so changes to the shared dependency declarations apply here
as well as in CI.
