---
title: 'Assembly Theory Tools: A Package for doing Assembly Theory calculations'
tags:
  - Python
  - Assembly Theory
authors:
  - name: Louie Slocombe
    orcid: 0000-0000-0000-0000
    # equal-contrib: true
    corresponding: true 
    affiliation: 1 
  - name: Joseph Fedrow
    orcid: 0000-0000-0000-0000
    # equal-contrib: true 
    affiliation: 1
  - name: Estelle Janin
    orcid: 0000-0000-0000-0000
    # equal-contrib: true 
    affiliation: 1
  - name: Gage Siebert
  	orcid: 0000-0000-0000-0000
  	# equal-contrib: true 	
    affiliation: 1
  - name: Veronica Mierzejewski
  	orcid: 0000-0000-0000-0000	
  	# equal-contrib: true 
    affiliation: 1
  - name: Marina Fernandez-Ruz
  	orcid: 0000-0000-0000-0000	
  	# equal-contrib: true 
    affiliation: 2
affiliations:
 - name: Beyond Center for Fundamental Concepts in Science, ASU, United States
   index: 1
 - name: Centro De Astrobiologia-CAB, Spain
   index: 1
date: 26 June 2025
bibliography: paper.bib

# Summary

In the vastness of molecular space, where does structure and selection come from?
Assembly Theory can potentially explain selection and evolution at the molecular level 
by quantifying the shortest path of construction from basic chemical building blocks to
complex macro molecules. As a potential new measure of complexity, Assembly Theory can 
be applied to situations beyond just molecular construction, for example strings, 
minerals, and even memes.  

# Statement of need

`Assembly Theory Tools` (ATT) is a Python package for doing Assembly Theory calculations. 
`ATT` was designed to be used by both researchers working at the interface between 
complexity, evolution, and synthetic chemistry, and students in courses on complexity. 
Over the years, the initial code base for doing Assembly calculations has been 
continually refined, starting from the GO language, to C++, to now being wrapped within
Python for ease of use and exploration.

# Mathematics

The equation for Assembly (including both the copy number and assembly index) has the
following form:

$$A = \sum_{i=1}^{N} \exp\bigl(a_{i}\bigr)\,\left(\frac{n_{i}-1}{N_{\mathrm T}}\right,$$

where $N$ is the number of unique objects in the ensemble, $a_i$ is the assembly index 
(minimal number of assembly steps), $n_i$ is the copy number, and $N_{\mathrm T}$ is the 
total number of objects. In practice, only the assembly index is used when comparing 
theory to experiment. In addition to having functions for calculating the assembly index
for molecules `Assembly Theory Tools` also has functions for calculating the assembly 
of strings as well. Complimentary functions such as the assembly distance semi-metric and 
pathway visualizations are also included for ease of exploration and use. 

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements


# References