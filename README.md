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


# For Local Installation 
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
Best to make them strict
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
conda install conda-forge::numpy conda-forge::matplotlib conda-forge::networkx conda-forge::rdkit conda-forge::openbabel conda-forge::pyvis -y
```
The, install the ELIFE packages
```
pip install git+https://github.com/ELIFE-ASU/dagviz.git git+https://github.com/ELIFE-ASU/CFG.git
```
Then install AssemblyTheoryTools:
```
pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git@v1.1.0
```
When asked for a password, you will need your GitHub username AND 'personal access token' (found in developer settings in your GitHub settings).
See `https://stackoverflow.com/questions/2505096/clone-a-private-repository-github`

# For Development Installation
Follow the instructions for making a fresh environment. Then, install the requirements.
```
conda install conda-forge::numpy conda-forge::matplotlib conda-forge::networkx conda-forge::rdkit conda-forge::pyvis conda-forge::pytest -y
```
The, install the ELIFE packages
```
pip install git+https://github.com/ELIFE-ASU/dagviz.git git+https://github.com/ELIFE-ASU/CFG.git
```

Clone the repo using Git or GitKraken. Then, open your favorite IDE (Pycharm/VS Code) and the cloned repo.

# For HPC (SOL) Installation

Set up GitHub SHH keys.

Or use GitHub access tokens.

`unset SSH_ASKPASS`

Then do the following:
Load Mamba (SOL preferred) or Conda.
It is recommended that you start from a fresh environment to prevent issues.
```
module create -n ass_env python=3.12
```
Activate the new env.
```
module activate ass_env
```
Add channels in this order.
```
module config --env --add channels conda-forge
```
Best to make them strict
```
module config --set channel_priority true
```
To check your updated channel list, run:
```
module config --show channels
```
Make sure to upgrade the conda env to force the channel priority.
```
module update conda --all -y
```
Install all the dependencies.
```
mamba install conda-forge::numpy conda-forge::matplotlib conda-forge::networkx conda-forge::rdkit conda-forge::pyvis -y
```
Install AssemblyTheoryTools.
```
pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git@v1.1.0
```

Once again, you will need your username AND 'personal access token' entered as your password.

When running on an HPC, you should run Python using the absolute path to the directory, for example: `srun $HOME/.conda/envs/myEnv/bin/python3`
