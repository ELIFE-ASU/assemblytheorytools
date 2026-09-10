# Installation

`assemblytheorytools` requires **Python 3.12 or newer**. It works best on
Unix-like systems; on Windows, use the Windows Subsystem for Linux.

## From PyPI

```bash
pip install assemblytheorytools
```

This pulls in every runtime dependency and the Rust `assembly-theory` wheel,
which is all the {doc}`quick start <index>` needs. The C++ calculator is not
distributed as a binary: the first calculation that needs it builds
assemblyCPP from source, which takes a few minutes and needs `git` and a C++20
compiler. Set `ASS_PATH` to use a build you already have, and see
[Configuration](configuration.md#the-c-calculator) for the details.

## From source

```bash
git clone https://github.com/ELIFE-ASU/assemblytheorytools.git
cd assemblytheorytools
pip install -e ".[dev,docs]"
```

The `dev` extra adds `pytest` and `pytest-cov`; the `docs` extra adds Sphinx,
MyST-NB and the theme used to build this site; the `notebooks` extra adds
JupyterLab for running the protocol notebooks. Omit any extra when it is not
needed.

## Conda environment

Starting from a fresh environment avoids dependency conflicts.

```bash
conda create -n ass_env python=3.13
conda activate ass_env
```

Add `conda-forge` and make the channel priority strict, otherwise the RDKit and
ASE builds can be resolved against incompatible channels:

```bash
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda config --show channels
```

Install the compiled dependencies through conda, then the rest through pip:

```bash
conda install numpy scipy matplotlib networkx pydot rdkit pyvis ase -y
pip install git+https://github.com/ELIFE-ASU/dagviz.git assemblycfg assembly-theory
pip install assemblytheorytools
```

For a development environment, add `pytest` to the `conda install` line and
clone the repository instead of installing from PyPI.

## HPC (SOL)

```bash
module load mamba/latest
mamba create -n ass_env -c conda-forge python=3.13
source activate ass_env
mamba install -c conda-forge numpy scipy matplotlib networkx rdkit pyvis ase -y
pip install assemblytheorytools
```

If the dependency install is killed for exceeding memory, split it into several
smaller `mamba install` commands.

On an HPC scheduler, invoke Python by absolute path so the job lands in the
right environment:

```bash
srun $HOME/.conda/envs/ass_env/bin/python3 my_script.py
```

## Optional: a faster assemblyCPP build

ATT's on-demand build is a plain portable release. assemblyCPP also ships CMake
presets for tuned and parallel builds, which are worth using for large
molecules. It needs only CMake 3.25 or newer, Ninja and a C++20 compiler — no
Boost.

```bash
git clone https://github.com/ELIFE-ASU/assemblycpp-v5.git
cd assemblycpp-v5
cmake --preset performance      # tuned for x86-64-v3
cmake --build --preset performance
export ASS_PATH=$PWD/build/performance/AssemblyCpp
```

`--preset release` builds a portable executable instead, and
`--preset parallel` adds OpenMP and MPI search. See the
[assemblycpp-v5 README](https://github.com/ELIFE-ASU/assemblycpp-v5) for the
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
