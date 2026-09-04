# Citing

`assemblytheorytools` is MIT licensed. If it contributes to work you publish,
please cite the papers the methods come from.

## Assembly theory

Sharma, A., Czégel, D., Lachmann, M., Kempes, C. P., Walker, S. I., & Cronin, L.
(2023). Assembly theory explains and quantifies selection and evolution.
*Nature*, 622(7982), 321–328.
[doi:10.1038/s41586-023-06600-9](https://doi.org/10.1038/s41586-023-06600-9)

## The assembly index algorithm

Seet, I., Patarroyo, K. Y., Siebert, G., Walker, S. I., & Cronin, L. (2024).
Rapid computation of the assembly index of molecular graphs. *arXiv preprint*
arXiv:2410.09100.
[doi:10.48550/arXiv.2410.09100](https://doi.org/10.48550/arXiv.2410.09100)

This describes the assemblyCPP calculator that ATT drives by default.

## The Rust calculator

Vimal, D., Parzych, G., Smith, O. M., Parkar, D., Bergen, H., Daymude, J. J.,
& Mathis, C. (2026). assembly-theory: Open, reproducible calculation of assembly
indices. *Journal of Open Source Software*, 11(117), 9318.
[doi:10.21105/joss.09318](https://doi.org/10.21105/joss.09318)

This describes the [assembly-theory](https://github.com/DaymudeLab/assembly-theory)
crate, reached through
{func}`~assemblytheorytools.assembly.calculate_assembly_index_rust` and the
other Rust-backed functions.

## Method-specific references

Cite these in addition when you use the corresponding part of the package:

* **Molecular weight against assembly index at scale** — Marshall, S. M. *et al.*
  (2021). Identifying molecules as biosignatures with assembly theory and mass
  spectrometry. *Nature Communications*, 12, 3033.
  [doi:10.1038/s41467-021-23258-x](https://doi.org/10.1038/s41467-021-23258-x)
* **Spectroscopic estimation of assembly index** — Jirasek, M. *et al.* (2024).
  Investigating and quantifying molecular complexity using assembly theory and
  spectroscopy. *ACS Central Science*, 10(5), 1054–1064.
  [doi:10.1021/acscentsci.4c00120](https://doi.org/10.1021/acscentsci.4c00120)
* **Ensemble assembly and the exploration ratio** — Jirasek, M., Sharma, A.,
  Wong, M., Munro, J., & Cronin, L. (2026). Quantifying the emergence of
  selection prior to biological evolution. *arXiv preprint* arXiv:2512.18752.
  [doi:10.48550/arXiv.2512.18752](https://doi.org/10.48550/arXiv.2512.18752)

  The source for {func}`~assemblytheorytools.assembly.exploration_ratio`, the
  union approximation of the joint assembly space behind
  {func}`~assemblytheorytools.assembly.joint_assembly_space`, and
  {doc}`Protocol 5 <examples/protocol_5>`.

## Datasets

Cite the data as well as the method when a workflow depends on an external
dataset:

* **Chemotion IR** — Jung, N., Tremouilhac, P., Punjabi, D., & Huang, P.-C.
  (2024). *Chemotion Repository - Data collection: FT-IR spectroscopy data
  (Chemotion IR)* [Data set]. Karlsruhe Institute of Technology.
  [doi:10.22000/OGoEQGlsZGElrgst](https://doi.org/10.22000/OGoEQGlsZGElrgst)

  The 4183 JCAMP-DX spectra behind
  {func}`~assemblytheorytools.tools_data.process_chemotion_ir_data` and
  {doc}`Protocol 3 <examples/protocol_3>`. Released under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), so
  redistributed or derived spectra must keep the attribution and the same
  licence. ATT neither bundles nor redistributes the archive — the protocol
  downloads it from the DOI.

## BibTeX

`att.bib` in the repository root holds BibTeX entries for these papers and for
the scientific Python stack ATT builds on — NumPy, SciPy, NetworkX, RDKit,
matplotlib and ASE. Please cite those too where your journal's conventions allow
it.
