# assemblytheorytools
A centralised set of tools for doing assembly theory calculations.

The code needs a compiled assemblyCPP in your path, put `export ASS_PATH=/data/grp_swalke10/asscpp/v5/asscpp_v5_recursive` in your submission script or in your `.bashrc`.

# Requirements
Make sure to load your conda environment. I would install them in this order:
- numpy `conda install numpy`
- matplotlib `conda install matplotlib`
- network x `pip install networkx[default]`
- rdkit `conda install -c conda-forge rdkit`
- pyvis `pip install pyvis`

# Install instructions
`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git`


# For Local Install:

Open a virtual environment (for example, using Pycharm)

`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git`

When asked for a password, you will need your GitHub username AND 'personal access token' (found in developer settings in your GitHub settings).


# For your SOL:

`unset SSH_ASKPASS`

`module load mamba/latest`

`source activate myEnv`

`mamba update -n myEnv`

`pip install git+https://github.com/ELIFE-ASU/assemblytheorytools.git`

Once again, you will need your username AND 'personal access token' entered as your password.

Using SOL is still new, so if any issues are encountered, please let us know~!
