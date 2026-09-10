# Installation

`assemblytheorytools` requires **Python 3.12 or newer**. It works best on
Unix-like systems; on Windows, use the Windows Subsystem for Linux.

## From PyPI

```bash
python -m pip install assemblytheorytools
```

This pulls in every runtime dependency and the Rust `assembly-theory` wheel,
which is all the {doc}`quick start <index>` needs. The C++ calculator is not
distributed as a binary: the first calculation that needs it builds
parallelassemblycpp from source, which takes a few minutes and needs `git` and a C++20
compiler. Set `ASS_PATH` to use a build you already have, and see
[Configuration](configuration.md#the-c-calculator) for the details.

## From source

```bash
git clone https://github.com/ELIFE-ASU/assemblytheorytools.git
cd assemblytheorytools
python -m pip install -e ".[dev,docs]"
```

The `dev` extra adds `pytest` and `pytest-cov`; the `docs` extra adds Sphinx,
MyST-NB and the theme used to build this site; the `notebooks` extra adds
JupyterLab for running the protocol notebooks. Omit any extra when it is not
needed. Dependency versions are declared in the repository's `pyproject.toml`.

To install the build and lint tools, use pip 25.1 or newer and the dependency
groups declared in the same file:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group build --group lint
python -m build
python -m twine check --strict dist/*
```

For documentation build systems that require a requirements file, run
`python -m pip install -r docs/requirements.txt` from the repository root. This
installs the package with its `docs` extra. Build the documentation with
`make -C docs strict`.

## Conda environment

The repository provides Conda environment files with Python, Git, a C++ compiler
and Cairo. They use only conda-forge and select Python 3.12–3.14, the versions
tested in CI. Python package dependencies are installed through pip from ATT's
package metadata.

After cloning the repository, run these commands from its root to install the
published package in a fresh environment:

```bash
conda env create -f build_tools/environment.yml
conda activate att_env
```

For an editable development installation with test, documentation and notebook
extras, use the development environment instead:

```bash
conda env create -f build_tools/environment_dev.yml
conda activate att_dev_env
python -m pip install --group build --group lint
```

## HPC (SOL)

On SOL, load Mamba and create the same environment from the repository root.
Module names and activation commands may differ on other HPC systems.

```bash
module load mamba/latest
mamba env create -f build_tools/environment.yml
source activate att_env
```

On an HPC scheduler, invoke Python by absolute path so the job lands in the
right environment:

```bash
srun "$HOME/.conda/envs/att_env/bin/python3" my_script.py
```

(optional-a-faster-assemblycpp-build)=
## Optional: a faster parallelassemblycpp build

ATT's on-demand build is a plain portable release. parallelassemblycpp also ships CMake
presets for tuned and parallel builds, which are worth using for large
molecules. It needs only CMake 3.25 or newer, Ninja and a C++20 compiler — no
Boost.

```bash
git clone https://github.com/ELIFE-ASU/parallelassemblycpp.git
cd parallelassemblycpp
cmake --preset performance      # tuned for x86-64-v3
cmake --build --preset performance
export ASS_PATH=$PWD/build/performance/ParallelAssemblyCpp
```

Older upstream revisions use the executable name `AssemblyCpp`; ATT accepts both
names. Its on-demand cache keeps the historical `AssemblyCpp` name.

`--preset release` builds a portable executable instead, and
`--preset parallel` adds OpenMP and MPI search. See the
[parallelassemblycpp README](https://github.com/ELIFE-ASU/parallelassemblycpp) for the
full list of presets and for its CC BY-NC 4.0 licence, which is more
restrictive than this package's MIT licence.

The same executable computes molecular, graph and string assembly indices. See
{doc}`configuration` for the full list of environment variables ATT reads.

## Optional: ORCA

Parts of {mod}`assemblytheorytools.tools_atoms` call
[ORCA](https://orcaforum.kofo.mpg.de/), a general-purpose quantum chemistry
package. ORCA is free for academic use but requires registration, and is only
needed for the energy and geometry-optimisation helpers — assembly index
calculations do not use it.

1. Register on the ORCA forum and open the *Downloads* section.
2. Download the build for your system, e.g. *ORCA 6.1.1, Linux, x86-64,
   shared-linked, .tar.xz*.
3. Extract it into an install directory such as `$HOME/orca_6_1_1`.
4. Point ATT at the executable:

   ```bash
   export ORCA_PATH=$HOME/orca_6_1_1/orca
   ```

## Verifying the install

```python
import assemblytheorytools as att

print(att.__version__)
print(att.calculate_assembly_index(att.smi_to_nx("CCO"), strip_hydrogen=True)[0])
```

This prints the installed version followed by `1`, the assembly index of the
hydrogen-stripped ethanol graph.
