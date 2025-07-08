---
title: 'Assembly Theory Tools: A Package for doing Assembly Theory calculations'
tags:
  - Python
  - Assembly Theory
  - Biochemistry
  - Astrobiology
authors:
  - name: Louie Slocombe
    orcid: 0000-0002-6986-5526
    affiliation: 1 
  - name: Joseph Fedrow
    orcid: 0009-0007-6908-2146
    affiliation: 1
  - name: Estelle Janin
    orcid: 0000-0003-0475-8479
    affiliation: 1
  - name: Gage Siebert
    orcid: 0000-0002-9390-4716
    affiliation: 1
  - name: Veronica Mierzejewski
    orcid: 0000-0002-6462-0001	
    affiliation: 1
  - name: Marina Fernandez-Ruz
    orcid: 0009-0006-9980-9230	
    affiliation: 2
  - name: Sara Walker
    orcid: 0000-0001-5779-2772
    corresponding: true 
    affiliation: 1
affiliations:
  - name: Beyond Center for Fundamental Concepts in Science, Arizona State University, United States
    index: 1
  - name: Centro De Astrobiologia-CAB, Spain
    index: 2
  - name: School of Earth and Space Exploration, Arizona State University, United States
  - index: 3
      
date: 30 June 2025
bibliography: paper.bib
---

# Summary

In the vastness of molecular space, where does structure and selection come from?
Assembly Theory can potentially explain selection and evolution at the molecular level 
by quantifying the shortest path of construction from basic chemical building blocks to
complex macro molecules. As a potential new measure of complexity, Assembly Theory can 
be applied to situations beyond just molecular construction, for example strings, 
minerals, and even memes.  

# Background

We characterize the structure of an assembly space with two things: (1) a set of elements we call virtual objects and (2) a set of relationships between virtual objects, which we call joining operations. Virtual objects are meant to represent entities in the universe, the most well-known instance being graphs representing molecules. We reserve the word object for the subset of virtual objects that represent observed entities with high copy number. Joining operations are meant to represent the ways in which these entities can combine. These operations are always of the form: two input virtual objects combine into one output virtual object.

An Assembly Space, denoted $\mathbb{A}=(\Omega,J)$, is defined by a set of virtual objects, $\Omega$, and a ternary relationship over that set $J:\Omega^3\to\{0,1\}$. We say there exists a joining operation that turns $x$ and $y$ into $z$ if and only if $J(x,y,z)=1$, where $x,y,z\in\Omega$. We require that $J(x,y,z)=J(y,x,z)~\forall x,y,z\in\Omega$. 
We call $u\in\Omega$ a unit if $\nexists x,y\in\Omega$ such that $J(x,y,u)=1$, and we denote the set of units by $\mathcal{U}$. We require that any $z\in\Omega$ can be reached from $\mathcal{U}$ by a finite sequence of joining operations. In other words, $\forall z\in\Omega$, $\exists \{z_1,z_2,...,z_{n-1},z_n\}$ such that $z=z_n$ and $\forall i\in[1,n],\exists x_i,y_i\in\mathcal{U}\cup\{z_1,z_2,...,z_{i-1}\}$ such that $J(x_i,y_i,z_i)=1$.

Units are virtual objects that cannot be the product of any joining operations, and they are typically the basis for the construction of the rest of $\Omega$. Nonempty $\Omega$ implies nonempty $\mathcal{U}$. Choosing how to model a system in assembly theory is what defines $\Omega$ and $J$.

An assembly path is a sequence of joining operations which act on units and products of previous joining operations. It can be represented with a sequence of tuples of the form $(x_i,y_i,z_i)$, for $i=1,...,n$ such that, $\forall i$ $J(x_i,y_i,z_i)=1$ and $x_i,y_i\in\mathcal{U}\cup\{z_1,z_2,...,z_{i-1}\}$. We require that each $z_i$ be distinct and we only consider assembly paths distinct up to a permutation of any $(x_i,y_i)$ pairs, since such permutations are trivial to perform. We call $n$ the path's length.

Then for any object, $x\in\Omega$, its assembly index, $a(x)$, is equal to the shortest length of an assembly path which produces $x$ from the units. In other words, it is the number of distinct joining operations required to build $x$ from the units. 

The equation for Assembly (including both the copy number and assembly index) has the
following form:

$$A = \sum_{i=1}^{N} \exp\bigl(a_{i}\bigr)\,\left(\frac{n_{i}-1}{N_{\mathrm T}}\right,$$

where $N$ is the number of unique objects in the ensemble, $a_i$ is the assembly index 
(minimal number of assembly steps), $n_i$ is the copy number, and $N_{\mathrm T}$ is the 
total number of objects. In practice, only the assembly index is used when comparing 
theory to experiment. In addition to having functions for calculating the assembly index
for molecules `Assembly Theory Tools` also has functions for calculating the assembly 
of strings as well. Complementary functions such as the assembly distance semi-metric and 
pathway visualisations are also included for ease of exploration and use. 

Assembly theory tools supports calculations in three distinct assembly spaces.

**Directed String Assembly Space**

Directed string assembly space is defined with respect to some finite set of characters, sometimes called an alphabet. It is of interest in the study of language and polymer chains. The virtual objects, $\Omega$, in directed string assembly are the finite strings which can be formed from some given alphabet. The joining operations are defined by: $J(x,y,z)=1$ if and only if $z$ can be formed by concatenating $x$ with $y$. From these definitions, it follows that the set of units is the alphabet.

**Undirected String Assembly Space**

Undirected string assembly space differs from its directed cousin only in its definition of virtual objects. It is still defined with respect to some finite set of characters, sometimes called an alphabet. It is also of interest to the study of polymer chains. The virtual objects, $\Omega$, in undirected string assembly are the finite strings which can be formed from some given alphabet, where we identify each string with its reversal. For example, the undirected strings "ab" and "ba" are the same virtual object. The joining operations are defined by: $J(x,y,z)=1$ if and only if $z$ can be formed by concatenating $x$ with $y$. From these definitions, it follows that the set of units is the alphabet.

**Molecular Assembly Space**

Molecular assembly space is defined with respect to a set of possible bonds: $"C=O"$, $"C-N"$, etc. This is applied to chemistry and is perhaps the best known specific assembly space. There are many ways one could define a molecular assembly space, but to focus on the topological structure of molecules, we use the following definition. The virtual objects, $\Omega$, in molecular assembly space represent molecules with connected simple graphs with edge and vertex colors, also known as molecular graphs. Colors are just a general term for labels on parts of a graph. Edge colors are things like "single" or "double", describing the bond types. Vertex colors are things like "C" or "O" describing the atoms. 

In molecular assembly space, the joining operations are defined by: $J(x,y,z)=1$ if and only if, there exists a map from $\{x,y\}\to z$ which identifies a non-trivial subset of the vertices of $x$ and $y$ to form $z$. Only vertices with the same color may be identified. No two vertices from $x$ may be identified with each other, nor can any two vertices from $y$ be identified with each other. It follows from the fact that $z\in\Omega$ that $z$ has no multi-edges implying that no pairs of adjacent\footnote{Two vertices are adjacent if they share an edge.} vertices from $x$ and $y$ may be identified. It also follows that the units are single bonds, which are represented with the set of molecular graphs with a single edge.

# Statement of Need

`AssemblyTheoryTools` (ATT) is a Python package for doing Assembly Theory calculations. 
ATT was designed to be used by both researchers working at the interface between 
complexity, evolution, and synthetic chemistry, and students in courses on complexity. 
Over the years, the initial code base for doing Assembly calculations has been 
continually refined, starting from the GO language, to C++, to now being wrapped within
Python for ease of use and exploration.

ATT is the centralised package for doing Assembly Theory calculations, 
it is primarily a tools ecosystem that interfaces with other packages such as CFG, assemblyCPP, and CFGgraph.

# Functionality and Usage Examples

# Tests and Benchmarks

# Availability and Governance

`AssemblyTheoryTools` source code and documentation are available on [GitHub](https://github.com/ELIFE-ASU/assemblytheorytools). The package is licensed under the MIT license.
External feedback and code contributions are handled through the usual Issues and Pull Request interfaces; guidelines for contributions are listed in `HACKING.md`.
The project's maintainers (initially Louie Slocombe and Gage Siebert) will govern it using the committee model: high-level decisions about the project's direction require maintainer consensus, major code changes require majority approval, hotfixes and patches require at least one approval, new maintainers may be added by unanimous decision of the existing maintainers, and existing maintainers may step down with advance notice.



# Author Contributions

Louie Slocombe, orchestration, development and conceptualisation.

Joey Fedrow, development, maintenance, documentation.

Estelle Janin, bonding and joint assembly index calculations,

Gage Siebert, string assembly index calculations and CFG integration.

Keith Patarroyo, assembly path reconstruction and visualisation.

Ian Seet, joining operations index calculations.

Sebastian Pagel, reassembly calculations and visualisation.

Veronica Mierzejewski, integration of reassembly calculations.

Marina Fernandez-Ruz, visualisation and circle plots.

# Acknowledgements

Louie Slocombe acknowledges support from the Beyond Center for Fundamental Concepts in Science at Arizona State University.

# References
