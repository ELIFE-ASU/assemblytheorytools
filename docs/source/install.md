# Installation

`assemblytheorytools` requires **Python 3.12 or newer**. It works best on
Unix-like systems; on Windows, use the Windows Subsystem for Linux.

## From PyPI

```bash
pip install assemblytheorytools
```

This pulls in every runtime dependency and the Rust `assembly-theory` wheel. On
Linux x86-64, the package also includes precompiled C++ calculators, which is
all that is needed for the {doc}`quick start <index>`. Other platforms need a
source build and must configure `ASS_PATH` for graph/molecule calculations and
`ASS_STR_PATH` for directed strings (a compatible combined build may serve
both); see
[Configuration](configuration.md#bundled-binaries).

## From source

```bash
git clone https://github.com/ELIFE-ASU/assemblytheorytools.git
cd assemblytheorytools
pip install -e ".[dev,docs]"
```

The `dev` extra adds `pytest` and `pytest-cov`; the `docs` extra adds Sphinx and
the theme used to build this site. Omit either extra when it is not needed.

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

## Optional: a faster assemblyCPP with Intel oneAPI

The bundled calculator is a generic static build. On Intel hardware, compiling
`assemblycpp` yourself with the oneAPI compiler is significantly faster.

Install the [oneAPI DPC++/C++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html?operatingsystem=linux&distribution-linux=offline)
and source its environment:

```bash
bash ./intel-dpcpp-cpp-compiler-2025.0.4.20_offline.sh
source ~/intel/oneapi/setvars.sh
```

Fetch Boost and the assemblyCPP sources:

```bash
wget https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz
tar -xvzf boost_1_89_0.tar.gz && rm -f boost_1_89_0.tar.gz
git clone --branch script https://github.com/LouieSlocombe/assemblycpp-v5.git
```

Compile, then point ATT at the result:

```bash
cd assemblycpp-v5/v5/
icpx main.cpp -o asscpp -I $HOME/boost_1_89_0/ -O3 -ipo -xHost -ffast-math -qopt-zmm-usage=high -fno-alias
export ASS_PATH=$HOME/assemblycpp-v5/v5/asscpp
```

See {doc}`configuration` for the full list of environment variables ATT reads.

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
