# AssemblyTheoryTools
A centralised set of tools for doing assembly theory calculations.

The code needs a compiled assemblyCPP in your path with the `ASS_PATH` environmental variable accessible by Python. 

For example, put `export ASS_PATH=/data/grp_swalke10/asscpp/v5_boost/asscpp_v5_boost_recursive` in your submission script or your `.bashrc`.

Currently supports and connects to:
- Molecules, including ionic bonded structures.
- Strings [https://github.com/ELIFE-ASU/CFG](https://github.com/ELIFE-ASU/CFG).
- Minerals see [https://github.com/ELIFE-ASU/ProteinAssembly](https://github.com/ELIFE-ASU/Mineral-evo).
- For Proteins see [`https://github.com/ELIFE-ASU/ProteinAssembly`](https://github.com/ELIFE-ASU/ProteinAssembly).

# Requirements
Make sure to load your conda environment. I would install them in this order:
- numpy `conda install numpy`
- matplotlib `conda install matplotlib`
- networkx `conda install anaconda::networkx`
- rdkit `conda install conda-forge::rdkit`
- pyvis `conda install conda-forge::pyvis`

# For Local Install:
Install the requirements above (conda recommended). In one go:

`conda install numpy matplotlib anaconda::networkx conda-forge::rdkit conda-forge::pyvis`

Then install assemblytheorytools:

`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git`

When asked for a password, you will need your GitHub username AND 'personal access token' (found in developer settings in your GitHub settings).
See `https://stackoverflow.com/questions/2505096/clone-a-private-repository-github`

# For your SOL:

`unset SSH_ASKPASS`

`module load mamba/latest`

`source activate myEnv`

`mamba install numpy matplotlib anaconda::networkx conda-forge::rdkit conda-forge::pyvis`

`mamba update --all -y`

`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git`

Once again, you will need your username AND 'personal access token' entered as your password.

I suggest running Python using the absolute path to the directory `srun $HOME/.conda/envs/myEnv/bin/python3`
