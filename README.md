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

# For Local Installation 
Install the requirements (conda recommended). In one go:

`conda install conda-forge::numpy conda-forge::matplotlib anaconda::networkx conda-forge::rdkit conda-forge::pyvis`

Then install AssemblyTheoryTools:

`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git@1.0.0`

When asked for a password, you will need your GitHub username AND 'personal access token' (found in developer settings in your GitHub settings).
See `https://stackoverflow.com/questions/2505096/clone-a-private-repository-github`

# For Development Installation
Install the requirements (conda recommended). In one go:

`conda install conda-forge::numpy conda-forge::matplotlib anaconda::networkx conda-forge::rdkit conda-forge::pyvis anaconda::pytest`

Clone the repo using Git or GitKraken. Then, open your favorite IDE (Pycharm/VS Code) and open the cloned repo.

# For HPC (SOL) Installation

Set up GitHub SHH keys.

Or use GitHub access tokens.

`unset SSH_ASKPASS`

Then do the following:

`module load mamba/latest`

`source activate myEnv`

`mamba install conda-forge::numpy conda-forge::matplotlib anaconda::networkx conda-forge::rdkit conda-forge::pyvis`

`mamba update --all -y`

`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git@1.0.0`

Once again, you will need your username AND 'personal access token' entered as your password.

When running on an HPC you should run Python using the absolute path to the directory, for example: `srun $HOME/.conda/envs/myEnv/bin/python3`
