from setuptools import setup, find_packages

setup(
    name='assemblytheorytools',
    version='1.0.0',
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
        'ase>=3.23.0',
        'numpy>=2.1.3',
        'matplotlib>=3.9.2',
        'networkx>=3.4.2',
        'rdkit>=2024.03.5',
        'ipython>=8.30.0',
        'pyvis>=0.3.2',
        'CFG @ git+https://github.com/ELIFE-ASU/CFG.git',
        'dagviz @ git+https://github.com/ELIFE-ASU/dagviz.git',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
)
