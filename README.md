
<img width="1931" height="665" alt="Untitled (5)" src="https://github.com/user-attachments/assets/26adaf9c-3841-4ef8-a0dc-61dabf923718" />




# AssemblyTheoryTools
A centralised set of tools for doing assembly theory calculations written in Python.
The aim is that this package provides a platform to do calculations that works out of the box. 
We currently interface with AssemblyCPP and this package comes with a precompiled version.
This version works on unix based systems and to use this package, it is strongly suggested that
you use the Linux subsystem if you are using Windows.

AssemblyTheoryTools (ATT) is a Python package designed to facilitate assembly theory calculations 
across various domains, including molecules, minerals, and proteins. It provides a unified interface 
to perform complex assembly theory computations, leveraging the power of AssemblyCPP.

Assembly theory is a framework that aims to quantify the complexity of 
objects by considering the minimal number of steps needed to assemble them 
from their fundamental building blocks. It essentially treats objects not as simple 
particles, but as entities defined by their possible formation histories, and it 
provides a way to measure how much selection was required to produce a given 
object or set of objects. 

Currently, ATT supports and connects to:

- Molecules, including ionic bonded structures.
- Approximate fast methods [https://github.com/ELIFE-ASU/CFGgraph](https://github.com/ELIFE-ASU/CFGgraph).
- Strings [https://github.com/ELIFE-ASU/CFG](https://github.com/ELIFE-ASU/CFG).

If you find this package useful, please cite the following papers: 
Sharma _et al._ 2023 and Seet _et al._ 2024 which can be found in paper.bib.

# Getting started
Checkout the requirements and installation instructions below.

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
However, if you want to use your own version, you can do so by setting the `ASS_PATH` environmental 
variable to the path of your AssemblyCPP installation.
For example, put `export ASS_PATH=/home/user/asscpp` in your submission 
script or your `.bashrc`. 
For compilation instructions to make your own version from source checkout AssemblyCPP for instructions.

## Simple example

For most use cases the general calculation can be 
exposed via the `calculate_assembly_index` function.

Here is a simple example for Caffeine. First, bring up a terminal 
and activate your conda or pip environment where you installed ATT. Type in:
```
python3
```
In Python, first import the package:
```
import assemblytheorytools as att
```
Next, there are several ways to define a system. In this example case we are 
going to use a smiles string which corresponds to Caffeine.
```
smi = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
```
Next we must convert our smiles string into a molecular graph
```
graph = att.smi_to_nx(smi)
```
We are now ready to calculate the assembly index, to do this we will use the `calculate_assembly_index` function. 
We will also get the virtual objects and the assembly path.
```
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
```
Here, the `ai` integer representing the assembly index, 
`virt_obj` contains the virtual objects along the assembly path, 
which is stored as a dictionary where the keys are the types of virtual objects
and the values are the corresponding molecular graphs.
and `pathway` contains the assembly pathway used to calculate the assembly index.
The `pathway` is a directional graph where each node represents a virtual object,
and each edge represents a joining operation that combines input virtual objects 
into one output virtual object.

Lets convert the virtual objects along the assembly path into smiles strings.
```
smi_out = att.get_mol_pathway_to_smi(virt_obj)
```
We should be able to print the results:
```
print(f"Assembly index: {ai}", flush=True)
print(f"virt_obj: {smi_out}", flush=True)
```
We should see the output:
```
Assembly index: 9
virt_obj: 
{'file_graph': ['Cn1c(=O)c2c(ncn2C)n(C)c1=O'], 
 'remnant': ['CN(C=O)CN'], 
 'duplicates': ['CN(C)C=O', 'CN(C)C'], 
 'removed_edges': ['C=N', 'C=CC']}
```

# Contributions

Contributions of all kinds—bug reports, feature suggestions, code improvements, and documentation updates - are welcome!
Please follow standard Python practices, write clear commit messages, and ensure all code is well-documented and tested.
To contribute, branch the repo, make changes, and submit a pull request.

Contribution check-list:

- Code must be packaged into reusable functions or classes.
- Code must use current tooling.
- Code must have documentation.
- Code must have at least one passing test.

# Contributors

Louie Slocombe, orchestration, development and conceptualisation.

Joey Fedrow, development, maintenance, and documentation.

Estelle Janin, bonding and joint assembly index calculations,

Gage Siebert, string assembly index calculations and CFG integration.

Keith Patarroyo, assembly path reconstruction and visualisation.

Ian Seet, joining operations index calculations.

Sebastian Pagel, reassembly calculations and visualisation.

Veronica Mierzejewski, integration of reassembly calculations.

Marina Fernandez-Ruz, visualisation and circle plots.

# Full installation instructions

<details>
<summary>Local</summary>
<br>

## Fresh environment

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

## Install the requirements

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

Clone the repo using Git or GitKraken. Then, open your favorite IDE (Pycharm/VS Code) and the cloned repo.

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
