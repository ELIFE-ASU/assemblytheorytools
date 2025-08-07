
<img width="2096" height="934" alt="Frame - 1 (2)" src="https://github.com/user-attachments/assets/4cc72e01-3ea4-4c0e-abd8-ba1af100b79b" />

# 🧪 `AssemblyTheoryTools` <!-- [![Stars](https://img.shields.io/github/stars/ELIFE-ASU/assemblytheorytools.svg?style=social&maxAge=3600&label=Star)](https://github.com/ELIFE-ASU/assemblytheorytools/stargazers)-->
A centralised set of tools for doing assembly theory calculations [\[1\]](#ref1) written in Python.

## 🗺️ Overview
The aim is that this package provides a platform to do assembly theory calculations that work out of the box. 
We currently interface with [C++](https://github.com/croningp/assemblycpp-v5) [\[2\]](#ref2) and [Rust](https://github.com/DaymudeLab/assembly-theory) [\[3\]](#ref3) assembly calculators, and this package comes with precompiled versions of both.
This version works best on Unix-based systems, and to use this package, it is strongly suggested that you use the Linux subsystem if you are using Windows.

AssemblyTheoryTools (ATT) is a Python package designed to facilitate assembly theory calculations 
across various domains. It provides a unified interface to perform complex assembly theory computations, 
leveraging the power of the underlying assembly calculators.

Assembly theory is a framework that aims to quantify the complexity of 
objects by considering the minimal number of steps needed to assemble them 
from their fundamental building blocks. It essentially treats objects not as simple 
particles, but as entities defined by their possible formation histories, and it 
provides a way to measure how much selection was required to produce a given 
object or set of objects. 

Currently, ATT supports and connects to:

- General undirected graphs via NetworkX.
- Molecules via RDKit.
- Directed and undirected strings.
- Approximate fast methods [CFGs](https://github.com/ELIFE-ASU/CFG).

If you find this package useful, please cite the following papers: 
Sharma _et al._ 2023 [\[1\]](#ref1) and Seet _et al._ 2024 [\[2\]](#ref2), found in the paper.bib.

## 🔧 Installing
Check out the requirements and installation instructions below.

The simplest way to install ATT is to use pip, which is the recommended package manager for Python. Installation is as simple as,
```
pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git
```
When asked for a password, you will need your GitHub username and 'personal access token' (found in developer settings
in your GitHub settings).
See `https://stackoverflow.com/questions/2505096/clone-a-private-repository-github`
This is because the package is private and requires authentication to access.
Further instructions can be found on this [GitHub page](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

The code needs a compiled assemblyCPP, which is included in this package by default. 
However, if you want to use your version, you can set the `ASS_PATH` environmental 
variable to the path of your AssemblyCPP installation.
For example, put `export ASS_PATH=/home/user/asscpp` in your submission 
script or your `.bashrc`. 
For compilation instructions to make your version from source, check out AssemblyCPP for instructions.

## 💡 Example

For most use cases, the general calculation can be 
exposed via the `calculate_assembly_index` function.

Here is a simple example for Caffeine. First, bring up a terminal 
and activate the conda or pip environment where you installed ATT. Type in:
```
python3
```
In Python, first import the package:
```
import assemblytheorytools as att
```
Next, there are several ways to define a system. In this example case, we are 
going to use a SMILES string which corresponds to Caffeine.
```
smi = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
```
Next, we must convert our SMILES string into a molecular graph.
```
graph = att.smi_to_nx(smi)
```
We are now ready to calculate the assembly index using the `calculate_assembly_index` function. 
We will also get the virtual objects and the assembly path.
```
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
```
Here, the `ai` integer represents the assembly index, 
`virt_obj` contains the virtual objects along the assembly path.
and `pathway` contains the assembly pathway used to calculate the assembly index.
The `pathway` is a directional graph where each node represents a virtual object,
and each edge represents a joining operation that combines input virtual objects 
into one output virtual object.

We should be able to print the results:
```
print(f"Assembly index: {ai}", flush=True)
print(f"virt_obj: {virt_obj}", flush=True)
```
We should see the output:
```
Assembly index: 9
virt_obj: ['C=N', 'C=CC', 'CN(C)C=O', 'CN(C)C', 'CN(C=O)CN','Cn1c(=O)c2c(ncn2C)n(C)c1=O']
```
## 💭 Feedback
### ⚠️ Issue Tracker
Found a bug? Have an enhancement request? Head over to the [GitHub issue
tracker](https://github.com/ELIFE-ASU/assemblytheorytools/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as possible about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions of all kinds—bug reports, feature suggestions, code improvements, and documentation updates - are welcome! See
[`CONTRIBUTING.md`](https://github.com/ELIFE-ASU/assemblytheorytools/blob/main/CONTRIBUTING.md)
for more details.

## 👥 Contributors

Louie Slocombe, orchestration, development and conceptualisation.

Joey Fedrow, development, maintenance, and documentation.

Estelle Janin, bonding and joint assembly index calculations,

Gage Siebert, string assembly index calculations and CFG integration.

Keith Patarroyo, assembly path reconstruction and visualisation.

Ian Seet, joining operations index calculations.

Sebastian Pagel, reassembly calculations and visualisation.

Veronica Mierzejewski, integration of reassembly calculations.

Marina Fernandez-Ruz, visualisation and circle plots.

## ⚖️ License
MIT License. We just ask that you cite the relevant papers, please!

## 📚 References
- <a id="ref1">\[1\]</a> Sharma, A., Czégel, D., Lachmann, M., Kempes, C. P., Walker, S. I., & Cronin, L. (2023). Assembly theory explains and quantifies selection and evolution. Nature, 622(7982), 321-328. [doi:10.1038/s41586-023-06600-9](https://doi.org/10.1038/s41586-023-06600-9).
- <a id="ref2">\[2\]</a> Seet, I., Patarroyo, K. Y., Siebert, G., Walker, S. I., & Cronin, L. (2024). Rapid computation of the assembly index of molecular graphs. arXiv preprint arXiv:2410.09100. [doi:10.48550/arXiv.2410.09100](
https://doi.org/10.48550/arXiv.2410.09100).
- <a id="ref3">\[3\]</a> https://github.com/DaymudeLab/assembly-theory
## 🛠️ Full installation instructions

<details>
<summary>Local</summary>
<br>

### Fresh environment

It is recommended that you start from a fresh environment to prevent issues.

```
conda create -n ass_env python=3.12
```

Activate the new env.

```
conda activate ass_env
```

Add channels in this order.

```
conda config --env --add channels conda-forge
```

Best to make the channels strict to prevent conflicts

```
conda config --set channel_priority true
```

To check your updated channel list, run:

```
conda config --show channels
```

Make sure to upgrade the conda env to force the channel priority.

```
conda update conda --all -y
```

#### Install the requirements

```
conda install numpy scipy matplotlib networkx rdkit pyvis ase -y
```

Then, install the ELIFE packages

```
pip install git+https://github.com/ELIFE-ASU/dagviz.git git+https://github.com/ELIFE-ASU/CFG.git
```

Then, install AssemblyTheoryTools:

```
pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git
```

When asked for a password, you will need your GitHub username AND 'personal access token' (found in developer settings
in your GitHub settings).
See `https://stackoverflow.com/questions/2505096/clone-a-private-repository-github`

</details>

<details>
<summary>Development</summary>
<br>

It is recommended that you start from a fresh environment to prevent issues.

```
conda create -n ass_env python=3.12
```

Activate the new environment.

```
conda activate ass_env
```

Add channels in this order.

```
conda config --env --add channels conda-forge
```

Best to make them strict

```
conda config --set channel_priority true
```

Install the requirements.

```
conda install numpy scipy matplotlib networkx rdkit pyvis ase pytest -y
```

Then, install the ELIFE packages

```
pip install git+https://github.com/ELIFE-ASU/dagviz.git git+https://github.com/ELIFE-ASU/CFG.git
```

Clone the repo using Git or GitKraken. Then, open your favourite IDE (Pycharm/VS Code) and the cloned repo.

</details>

<details>
<summary>HPC-SOL</summary>
<br>

Set up GitHub SSH keys, or use GitHub access tokens. On the cluster, you use this command:

```
unset SSH_ASKPASS
```

Load Mamba

```
module load mamba/latest
```

It is recommended that you start from a fresh environment to prevent issues.

```
mamba create -n ass_env -c conda-forge python=3.12 
```

Activate the new env.

```
source activate ass_env
```

Install all the dependencies. If it kills, feel free to split the installs.

```
mamba install -c conda-forge numpy scipy matplotlib networkx rdkit pyvis ase -y
```

Install AssemblyTheoryTools.

```
pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git
```

Once again, you will need your username AND 'personal access token' entered as your password.

When running on an HPC, you should run Python using the absolute path to the directory, for example:
`srun $HOME/.conda/envs/myEnv/bin/python3`

</details>
