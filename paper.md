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
   index: 1
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

# Statement of Need

`Assembly Theory Tools` (ATT) is a Python package for doing Assembly Theory calculations. 
`ATT` was designed to be used by both researchers working at the interface between 
complexity, evolution, and synthetic chemistry, and students in courses on complexity. 
Over the years, the initial code base for doing Assembly calculations has been 
continually refined, starting from the GO language, to C++, to now being wrapped within
Python for ease of use and exploration.

# Functionality and Usage Examples

# Tests and Benchmarks

# Availability and Governance

`Assembly Theory Tools` source code and documentation are available on [GitHub](https://github.com/ELIFE-ASU/assemblytheorytools). The package is licensed under the MIT license.
External feedback and code contributions are handled through the usual Issues and Pull Request interfaces; guidelines for contributions are listed in `HACKING.md`.
The project's maintainers (initially Louie Slocombe and Gage Siebert) will govern it using the committee model: high-level decisions about the project's direction require maintainer consensus, major code changes require majority approval, hotfixes and patches require at least one approval, new maintainers may be added by unanimous decision of the existing maintainers, and existing maintainers may step down with advance notice.



# Author Contributions



# Acknowledgements



# References
