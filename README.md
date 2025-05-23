# AssemblyTheoryTools
A centralised set of tools for doing assembly theory calculations. To use this package, it is strongly suggested that you use the Linux subsystem if you are using Windows.

The code needs a compiled assemblyCPP in your path with the `ASS_PATH` environmental variable accessible by Python. 

For example, put `export ASS_PATH=/data/grp_swalke10/asscpp/v5_boost/asscpp_v5_boost_recursive` in your submission script or your `.bashrc`.

Currently supports and connects to:
- Molecules, including ionic bonded structures.
- Approximate fast methods [https://github.com/ELIFE-ASU/CFGgraph](https://github.com/ELIFE-ASU/CFGgraph).
- Strings [https://github.com/ELIFE-ASU/CFG](https://github.com/ELIFE-ASU/CFG).
- Minerals see [https://github.com/ELIFE-ASU/Mineral-evo](https://github.com/ELIFE-ASU/Mineral-evo).
- For Proteins see [https://github.com/ELIFE-ASU/ProteinAssembly](https://github.com/ELIFE-ASU/ProteinAssembly).

# Contributions
Contributions of all kinds—bug reports, feature suggestions, code improvements, and documentation updates - are welcome! Please follow standard Python practices, write clear commit messages, and ensure all code is well-documented and tested. To contribute, branch the repo, make changes, and submit a pull request. 

# Contributors
Louie Slocombe, Joey Fedrow, Estelle Janin, Gage Siebert. TBC: Veronica Mierzejewski, Marina Fernandez-Ruz

# Installation instructions

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
conda install numpy scipy matplotlib networkx rdkit openbabel pyvis ase -y
```

Then, install the ELIFE packages

```
pip install git+https://github.com/ELIFE-ASU/dagviz.git git+https://github.com/ELIFE-ASU/CFG.git
```

Then, install AssemblyTheoryTools:

```
pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git
```

When asked for a password, you will need your GitHub username AND 'personal access token' (found in developer settings in your GitHub settings).
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
conda install numpy scipy matplotlib networkx rdkit openbabel pyvis ase pytest -y
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

When running on an HPC, you should run Python using the absolute path to the directory, for example: `srun $HOME/.conda/envs/myEnv/bin/python3`

</details>
