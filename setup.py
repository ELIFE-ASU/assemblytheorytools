from setuptools import setup, find_packages

setup(
    name='assemblytheorytools',
    version='0.2.0',
    author='Louie Slocombe',
    author_email='louies@hotmail.co.uk',
    description='A centralised set of tools for doing assembly theory calculations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ELIFE-ASU/assemblytheorytools',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    install_requires=[
        'ase',
        'numpy',
        'matplotlib',
        'networkx',
        'rdkit',
        'ipython',
        'pyvis',
        'CFG @ git+https://github.com/ELIFE-ASU/CFG.git',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
)
