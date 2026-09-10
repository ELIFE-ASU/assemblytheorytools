<img
  src="https://github.com/user-attachments/assets/4cc72e01-3ea4-4c0e-abd8-ba1af100b79b"
  alt="AssemblyTheoryTools banner"
  width="100%"
/>

# AssemblyTheoryTools

[![Documentation Status](https://readthedocs.org/projects/assemblytheorytools/badge/?version=latest)](https://assemblytheorytools.readthedocs.io/en/latest/)
[![Tests](https://github.com/ELIFE-ASU/assemblytheorytools/actions/workflows/tests.yml/badge.svg)](https://github.com/ELIFE-ASU/assemblytheorytools/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/assemblytheorytools.svg)](https://pypi.org/project/assemblytheorytools/)
[![Python versions](https://img.shields.io/pypi/pyversions/assemblytheorytools.svg)](https://pypi.org/project/assemblytheorytools/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/LICENSE)

AssemblyTheoryTools (ATT) provides a unified Python interface for assembly-index calculations across molecules, strings,
and arbitrary graphs.

[Documentation](https://assemblytheorytools.readthedocs.io/) ·
[Examples](https://github.com/ELIFE-ASU/assemblytheorytools/tree/main/examples) ·
[API reference](https://assemblytheorytools.readthedocs.io/en/latest/modules.html) ·
[PyPI](https://pypi.org/project/assemblytheorytools/) ·
[Issues](https://github.com/ELIFE-ASU/assemblytheorytools/issues) ·
[Releases](https://github.com/ELIFE-ASU/assemblytheorytools/releases)

<details>
<summary><strong>What is assembly theory?</strong></summary>

Assembly theory quantifies the complexity of an object by the smallest number of joining steps needed to build it from
elementary parts, while allowing previously created intermediates to be reused. The reuse rule captures internal
structure and repetition rather than size alone.

For molecules, the elementary parts are bonds and the calculation is performed on the molecular graph. ATT exposes the
[parallelassemblycpp](https://github.com/ELIFE-ASU/parallelassemblycpp) C++ calculator, the
[assembly-theory](https://github.com/DaymudeLab/assembly-theory) Rust calculator, and
[assemblycfg](https://github.com/ELIFE-ASU/assemblycfg) for fast approximate string calculations through one Python
package.

See the [concepts guide](https://assemblytheorytools.readthedocs.io/en/latest/concepts.html) and
[theory overview](https://assemblytheorytools.readthedocs.io/en/latest/theory.html) for more background.

</details>

## Quick start

ATT requires **Python 3.12 or newer**. Install the current release from PyPI:

```bash
python -m pip install assemblytheorytools
```

> **Platform note:** The C++ calculator is not distributed as a binary. The first calculation that needs it builds
> [parallelassemblycpp](https://github.com/ELIFE-ASU/parallelassemblycpp) from source into `~/.cache/assemblytheorytools`, which
> takes a few minutes and needs a C++20 compiler. To use a build you already have, set `ASS_PATH` instead; see
> **Use your own parallelassemblycpp build** below.

Calculate and plot the assembly pathway for caffeine:

```python
import matplotlib.pyplot as plt

import assemblytheorytools as att

smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
graph = att.smi_to_nx(smi)

ai, virtual_objects, pathway = att.calculate_assembly_index(
    graph,
    strip_hydrogen=True,
)

print(f"Assembly index: {ai}")

fig, ax = att.plot_pathway(pathway, plot_type="graph")
plt.show()
```

```text
Assembly index: 9
```

![Assembly pathway for caffeine](https://raw.githubusercontent.com/ELIFE-ASU/assemblytheorytools/main/readme_example.png)

<details>
<summary><strong>Understanding the result</strong></summary>

- `ai` is the assembly index.
- `virtual_objects` contains reusable intermediates found along the pathway. The collection is unordered; do not rely on
  positional order.
- `pathway` is a NetworkX `DiGraph`: its nodes are virtual objects and its directed edges are joining operations.

Convert the virtual-object graphs back to SMILES with:

```python
virtual_smiles = [
    att.nx_to_smi(obj, add_hydrogens=False)
    for obj in virtual_objects
]
```

Most published molecular assembly indices exclude hydrogens. Use `strip_hydrogen=True` when comparing against those
values. ATT strips a copy, leaving the original graph unchanged.

</details>

## Quick start reference

Which function computes which quantity. The
[route map](https://assemblytheorytools.readthedocs.io/en/latest/route_map.html) carries the full table — every
quantity ATT computes, with its inputs, its outputs, and what it is used for.

| Quantity | Function | Input | Output |
| --- | --- | --- | --- |
| Assembly index | `calculate_assembly_index` | NetworkX graph or RDKit `Mol` | Index, virtual objects, pathway |
| Assembly index without the pathway | `calculate_assembly_index_rust` | NetworkX graph or RDKit `Mol` | Index (hydrogens always stripped) |
| String assembly index | `calculate_string_assembly_index` | String or list of strings | Index, virtual objects, pathway |
| Joint assembly index | `calculate_assembly_index` on a joined graph | Graphs merged with `join_graphs` | Index for the whole set |
| Shared-assembly score | `calculate_assembly_index_similarity` | List of graphs | Score; 0 to 1 for a pair |
| Semi-metric distance | `calculate_assembly_index_semi_metric` | Two graphs | Distance; negative means cheaper together |
| Assembly `A` | `calculate_assembly` | Graphs and their copy numbers | Ensemble assembly value |
| Assembly depth | `calculate_assembly_depth_rust` | NetworkX graph or RDKit `Mol` | Minimum depth under concurrent joins |
| Bounds | `calculate_assembly_index_upper_bound`, `calculate_assembly_index_lower_bound` | NetworkX graph or RDKit `Mol` | Instant bounds for screening |
| Many indices at once | `calculate_assembly_index_parallel` | List of graphs | Indices, virtual objects, pathways |
| Assembly index from tandem MS | `MAEstimator` | Fragmentation tree and molecular weight | Monte Carlo samples of MA |
| Assembly index from IR peaks | `estimate_ai_from_ir_peaks` | Peak counts and reference indices | Fitted model and predicted indices |
| Other complexity scores | `bertz_complexity`, `bottcher`, `wiener_index`, and more | RDKit `Mol` | Score, for comparison against the index |

## What ATT includes

- Exact assembly-index calculations for molecules, arbitrary labelled graphs, and directed or undirected strings.
- Default C++ and alternative Rust search interfaces, plus fast graph bounds and CFG-based string approximations.
- Joint assembly, parallel execution, pathway parsing, pathway visualisation, and alternative-path enumeration.
- Molecular complexity metrics, structure conversion, reassembly, crystal-cell, spectroscopy, and mass-spectrometry
  utilities.

<details>
<summary><strong>Calculator backends and important differences</strong></summary>

| Backend | Main interface | Best suited to | Result |
| --- | --- | --- | --- |
| parallelassemblycpp (C++) | `calculate_assembly_index` | Default molecule and graph calculations | Index, virtual objects, and pathway |
| assembly-theory (Rust) | `calculate_assembly_index_rust` | Fast molecular index calculations | Index |
| assembly-theory search (Rust) | `calculate_assembly_index_rust_search` | Search statistics, options, and supported pathway reconstruction | Structured search result |
| Graph bounds | `calculate_assembly_index_upper_bound` and `calculate_assembly_index_lower_bound` | Fast size-based estimates | Upper or lower bound |
| assemblycfg | String calculations with `mode="cfg"` | Fast approximate string calculations | Upper bound and pathway |

The Rust backend always strips hydrogens. For a meaningful comparison, compare it with
`calculate_assembly_index(..., strip_hydrogen=True)`.

The PyPI distribution ships no parallelassemblycpp binary. ATT checks `ASS_PATH`, then looks for `ParallelAssemblyCpp`
(or the older `AssemblyCpp`) on `PATH`, then in its own cache, and builds one from source if it finds none. A single executable covers molecules,
graphs and strings. See [configuration](https://assemblytheorytools.readthedocs.io/en/latest/configuration.html) for
all backend options and environment variables.

</details>

## Installation

The one-line PyPI install above resolves ATT's runtime dependencies. The authoritative dependency list and minimum
versions live in
[`pyproject.toml`](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/pyproject.toml).

<details>
<summary><strong>Install from source for development</strong></summary>

```bash
git clone https://github.com/ELIFE-ASU/assemblytheorytools.git
cd assemblytheorytools
python -m pip install --upgrade "pip>=25.1"
python -m pip install -e ".[dev]"
python -m pip install --group build --group lint
pytest
```

Install `.[docs]` instead of `.[dev]`, or install both extras, to build the documentation:

```bash
python -m pip install -e ".[dev,docs]"
make -C docs strict
```

See
[`CONTRIBUTING.md`](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/CONTRIBUTING.md)
for the full development workflow.

</details>

<details>
<summary><strong>Use a Conda environment</strong></summary>

From the root of a repository checkout, create an environment with Python, Git, a C++ compiler and Cairo. The environment
uses only conda-forge and Python 3.12–3.14; pip installs ATT's Python dependencies from its package metadata:

```bash
conda env create -f build_tools/environment.yml
conda activate att_env
```

For an editable development installation, use `build_tools/environment_dev.yml` and activate `att_dev_env` instead.
See [`build_tools/README.md`](build_tools/README.md) for the build tools and checks.

</details>

<details>
<summary><strong>Install on an HPC system (including SOL)</strong></summary>

Module names vary between systems. On SOL, run these commands from the root of a repository checkout:

```bash
module load mamba/latest
mamba env create -f build_tools/environment.yml
source activate att_env
```

When submitting a scheduled job, use the environment's Python executable explicitly:

```bash
srun "$HOME/.conda/envs/att_env/bin/python3" my_script.py
```

</details>

<details>
<summary><strong>Use your own parallelassemblycpp build</strong></summary>

ATT builds parallelassemblycpp on demand, but a build you control is worth having: it skips the wait on first use, and an
optimised build is faster on supported hardware. Build it once:

```bash
git clone https://github.com/ELIFE-ASU/parallelassemblycpp.git
cmake -S parallelassemblycpp -B parallelassemblycpp/build/release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build parallelassemblycpp/build/release --parallel
```

Then set `ASS_PATH` to the **full path of the executable**, not its containing directory:

```bash
export ASS_PATH=$PWD/parallelassemblycpp/build/release/ParallelAssemblyCpp
```

Older upstream revisions name this executable `AssemblyCpp`; ATT accepts both names and keeps the historical name
for its own cached build. The same executable computes molecular, graph and string assembly indices. `ASS_STR_PATH` is only needed to point
string calculations at a *different* build; it falls back to `ASS_PATH` when unset.

To build a specific revision on demand instead, set `ATT_ASSEMBLYCPP_REF` to a branch, tag or commit. For the
optimised and parallel build presets, see the
[installation guide](https://assemblytheorytools.readthedocs.io/en/latest/install.html#optional-a-faster-parallelassemblycpp-build).

> parallelassemblycpp is licensed CC BY-NC 4.0, which is more restrictive than this package's MIT licence. That is why ATT
> builds it on demand rather than distributing it.

</details>

<details>
<summary><strong>Optional: configure ORCA</strong></summary>

[ORCA](https://orcaforum.kofo.mpg.de/) is only required by energy and geometry-optimisation helpers in
`assemblytheorytools.tools_atoms`; ordinary assembly-index calculations do not use it. ORCA is free for academic use but
requires registration.

After downloading and extracting the appropriate ORCA build, point ATT at the executable:

```bash
export ORCA_PATH=/absolute/path/to/orca
```

See the [installation guide](https://assemblytheorytools.readthedocs.io/en/latest/install.html#optional-orca) for the
complete setup.

</details>

<details>
<summary><strong>Verify the installation</strong></summary>

```python
import assemblytheorytools as att

print(att.__version__)
print(
    att.calculate_assembly_index(
        att.smi_to_nx("CCO"),
        strip_hydrogen=True,
    )[0]
)
```

The second line prints `1`, the assembly index of hydrogen-stripped ethanol.

</details>

## Documentation and examples

| Resource | Description |
| --- | --- |
| [Route map](https://assemblytheorytools.readthedocs.io/en/latest/route_map.html) | Every ATT quantity with its inputs, outputs, and applications |
| [Concepts](https://assemblytheorytools.readthedocs.io/en/latest/concepts.html) | Assembly indices, virtual objects, pathways, joint assembly, and backends |
| [User guide](https://assemblytheorytools.readthedocs.io/en/latest/guide/index.html) | Molecules, strings, graphs, pathways, parallel runs, complexity, and mass spectrometry |
| [Runnable examples](https://github.com/ELIFE-ASU/assemblytheorytools/tree/main/examples) | Basic and advanced scripts included with the repository |
| [Published protocols](https://github.com/ELIFE-ASU/assemblytheorytools/tree/main/examples/protocols) | Jupyter notebooks reproducing published workflows end to end, committed with their outputs |
| [Configuration](https://assemblytheorytools.readthedocs.io/en/latest/configuration.html) | Environment variables, binaries, graph requirements, and search options |
| [API reference](https://assemblytheorytools.readthedocs.io/en/latest/modules.html) | Complete module and function documentation |

## Support and contributing

Found a bug or have a feature request? Open an issue in the
[GitHub tracker](https://github.com/ELIFE-ASU/assemblytheorytools/issues). Bug reports should include the ATT version,
Python version, operating system, and a minimal reproducible example.

Contributions are welcome. Read
[`CONTRIBUTING.md`](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/CONTRIBUTING.md)
before opening a pull request.

<details>
<summary><strong>Contributors and acknowledgements</strong></summary>

- Louie Slocombe — orchestration, development, and conceptualisation
- Gage Siebert — string assembly-index calculations and CFG integration
- Estelle Janin — bonding and joint assembly-index calculations
- Joey Fedrow — development, maintenance, and documentation
- Veronica Mierzejewski — integration of reassembly calculations
- Mohammadreza Shahjahan — branding and development
- Marina Fernandez-Ruz — visualisation and circle plots
- Sebastian Pagel — reassembly calculations and visualisation
- Amit Kahana — recursive MA integration
- Stuart Marshall — debugging and optimisation
- Ian Seet — joining-operations index calculations
- Keith Patarroyo — assembly-path reconstruction and visualisation
- Michael Jirasek — mass-spectrometry measurement pipeline
- Abhishek Sharma — administrative support
- Lee Cronin — concept, funding, and administrative support
- Sara Walker — concept, funding, and administrative support

</details>

## Citing

If ATT contributes to published work, cite the papers associated with the methods you use. The repository also includes
an [`att.bib`](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/att.bib) bibliography for ATT and its scientific
Python dependencies.

<details>
<summary><strong>References</strong></summary>

1. Sharma, A., Czégel, D., Lachmann, M., Kempes, C. P., Walker, S. I., & Cronin, L. (2023). Assembly theory explains
   and quantifies selection and evolution. *Nature*, 622(7982), 321–328.
   [doi:10.1038/s41586-023-06600-9](https://doi.org/10.1038/s41586-023-06600-9)
2. Seet, I., Patarroyo, K. Y., Siebert, G., Walker, S. I., & Cronin, L. (2024). Rapid computation of the assembly index
   of molecular graphs. *arXiv preprint*, arXiv:2410.09100.
   [doi:10.48550/arXiv.2410.09100](https://doi.org/10.48550/arXiv.2410.09100)
3. Vimal, D., Parzych, G., Smith, O. M., Parkar, D., Bergen, H., Daymude, J. J., & Mathis, C. (2026).
   assembly-theory: Open, reproducible calculation of assembly indices. *Journal of Open Source Software*, 11(117),
   9318. [doi:10.21105/joss.09318](https://doi.org/10.21105/joss.09318)

Method-specific references for spectroscopy and mass-spectrometry workflows are listed in the
[citing guide](https://assemblytheorytools.readthedocs.io/en/latest/citing.html).

</details>

## License

AssemblyTheoryTools is available under the
[MIT License](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/LICENSE).
