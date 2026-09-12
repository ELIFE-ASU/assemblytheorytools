# Citing

`assemblytheorytools` is MIT licensed. If it contributes to work you publish,
please cite the papers the methods come from. `att.bib` in the repository root
holds a BibTeX entry for everything on this page.

## Assembly theory

Sharma, A., Czégel, D., Lachmann, M., Kempes, C. P., Walker, S. I., & Cronin, L.
(2023). Assembly theory explains and quantifies selection and evolution.
*Nature*, 622(7982), 321–328.
[doi:10.1038/s41586-023-06600-9](https://doi.org/10.1038/s41586-023-06600-9)

## The calculators

ATT implements no search of its own. Cite the backend you actually ran.

### parallelassemblycpp (the default)

Seet, I., Patarroyo, K. Y., Siebert, G., Walker, S. I., & Cronin, L. (2025).
Rapid exploration of the assembly chemical space of molecular graphs.
*Journal of Chemical Information and Modeling*, 65(24), 13203–13214.
[doi:10.1021/acs.jcim.5c01964](https://doi.org/10.1021/acs.jcim.5c01964)

This describes the algorithm. ATT's default backend,
[parallelassemblycpp](https://github.com/ELIFE-ASU/parallelassemblycpp), is a
re-engineering of the paper's reference implementation and computes the same
index. It is reached through
{func}`~assemblytheorytools.assembly.calculate_assembly_index`.

### assembly-theory (Rust)

Vimal, D., Parzych, G., Smith, O. M., Parkar, D., Bergen, H., Daymude, J. J.,
& Mathis, C. (2026). assembly-theory: Open, reproducible calculation of assembly
indices. *Journal of Open Source Software*, 11(117), 9318.
[doi:10.21105/joss.09318](https://doi.org/10.21105/joss.09318)

This describes the [assembly-theory](https://github.com/DaymudeLab/assembly-theory)
crate, reached through
{func}`~assemblytheorytools.assembly.calculate_assembly_index_rust` and the
other Rust-backed functions.

### assemblycfg (approximate strings)

Siebert, G., Chowdhury, R., Slocombe, L., & Walker, S. (2026). *assemblycfg*
(v1.2.3) [Software]. Zenodo.
[doi:10.5281/zenodo.20562899](https://doi.org/10.5281/zenodo.20562899)

The RePair smallest-grammar upper bound reached with
`calculate_string_assembly_index(..., mode="cfg")`. There is no accompanying
paper; the project asks that the archived release be cited.

## Method-specific references

Cite these in addition when you use the corresponding part of the package:

* **Molecular weight against assembly index at scale** — Marshall, S. M.,
  Mathis, C., Carrick, E., Keenan, G., Cooper, G. J. T., Graham, H., Craven, M.,
  Gromski, P. S., Moore, D. G., Walker, S. I., & Cronin, L. (2021). Identifying
  molecules as biosignatures with assembly theory and mass spectrometry.
  *Nature Communications*, 12, 3033.
  [doi:10.1038/s41467-021-23258-x](https://doi.org/10.1038/s41467-021-23258-x)
* **Spectroscopic estimation of assembly index** — Jirasek, M. *et al.* (2024).
  Investigating and quantifying molecular complexity using assembly theory and
  spectroscopy. *ACS Central Science*, 10(5), 1054–1064.
  [doi:10.1021/acscentsci.4c00120](https://doi.org/10.1021/acscentsci.4c00120)

  The source for {mod}`assemblytheorytools.recursive_ma` and the infrared
  correlation in {mod}`assemblytheorytools.tools_data`. See
  {doc}`guide/mass_spectrometry`.
* **Assembly trees and fragment-pool reassembly** — Liu, Y., Mathis, C.,
  Bajczyk, M. D., Marshall, S. M., Wilbraham, L., & Cronin, L. (2021).
  Exploring and mapping chemical space with molecular assembly trees.
  *Science Advances*, 7(39), eabj2465.
  [doi:10.1126/sciadv.abj2465](https://doi.org/10.1126/sciadv.abj2465)
* **Generative reassembly and assembly depth** — Pagel, S., Sharma, A., &
  Cronin, L. (2026). Mapping evolution of molecules across biochemistry with
  assembly theory. *Journal of Chemical Information and Modeling*, 66(14),
  8239–8250.
  [doi:10.1021/acs.jcim.6c00939](https://doi.org/10.1021/acs.jcim.6c00939)

  The source for
  {class}`~assemblytheorytools.reassembler.MoleculeGenerationAssemblyPool` and
  for the depth definition behind
  {func}`~assemblytheorytools.assembly.calculate_assembly_depth_rust`. See
  {doc}`guide/reassembly`.
* **Ensemble assembly and the exploration ratio** — Jirasek, M., Sharma, A.,
  Wong, M., Munro, J., & Cronin, L. (2025). Quantifying the emergence of
  selection prior to biological evolution. *arXiv preprint* arXiv:2512.18752.
  [doi:10.48550/arXiv.2512.18752](https://doi.org/10.48550/arXiv.2512.18752)

  The source for {func}`~assemblytheorytools.assembly.exploration_ratio`, the
  union approximation of the joint assembly space behind
  {func}`~assemblytheorytools.assembly.joint_assembly_space`, and
  {doc}`Protocol 5 <examples/protocol_5>`.

## Complexity scores

{mod}`assemblytheorytools.complexity_scores` implements published measures for
comparison against the assembly index. Cite the one you report:

| Score | Reference |
| --- | --- |
| {func}`~assemblytheorytools.complexity_scores.bertz_complexity` | Bertz (1981), *JACS* 103(12), 3599–3601. [doi:10.1021/ja00402a071](https://doi.org/10.1021/ja00402a071) |
| {func}`~assemblytheorytools.complexity_scores.wiener_index` | Wiener (1947), *JACS* 69(1), 17–20. [doi:10.1021/ja01193a005](https://doi.org/10.1021/ja01193a005) |
| {func}`~assemblytheorytools.complexity_scores.randic_index` | Randić (1975), *JACS* 97(23), 6609–6615. [doi:10.1021/ja00856a001](https://doi.org/10.1021/ja00856a001) |
| {func}`~assemblytheorytools.complexity_scores.balaban_index` | Balaban (1982), *Chem. Phys. Lett.* 89(5), 399–404. [doi:10.1016/0009-2614(82)80009-2](https://doi.org/10.1016/0009-2614(82)80009-2) |
| {func}`~assemblytheorytools.complexity_scores.kirchhoff_index` | Klein & Randić (1993), *J. Math. Chem.* 12(1), 81–95. [doi:10.1007/BF01164627](https://doi.org/10.1007/BF01164627) |
| {func}`~assemblytheorytools.complexity_scores.bottcher` | Böttcher (2016), *JCIM* 56(3), 462–470. [doi:10.1021/acs.jcim.5b00723](https://doi.org/10.1021/acs.jcim.5b00723) |
| {func}`~assemblytheorytools.complexity_scores.proudfoot` | Proudfoot (2017), *Bioorg. Med. Chem. Lett.* 27(9), 2014–2017. [doi:10.1016/j.bmcl.2017.03.008](https://doi.org/10.1016/j.bmcl.2017.03.008) |
| {func}`~assemblytheorytools.complexity_scores.spacial_score` | Krzyzanowski, Pahl, Grigalunas & Waldmann (2023), *J. Med. Chem.* 66(18), 12739–12750. [doi:10.1021/acs.jmedchem.3c00689](https://doi.org/10.1021/acs.jmedchem.3c00689) |
| {func}`~assemblytheorytools.complexity_scores.mc1`, {func}`~assemblytheorytools.complexity_scores.mc2` | Buehler & Reymond (2025), *JCIM* 65(16), 8405–8410. [doi:10.1021/acs.jcim.5c00334](https://doi.org/10.1021/acs.jcim.5c00334) |
| {func}`~assemblytheorytools.complexity_scores.fcfp4` | Schuffenhauer *et al.* (2006), *JCIM* 46(2), 525–535. [doi:10.1021/ci0503558](https://doi.org/10.1021/ci0503558) |

The compression proxies
({func}`~assemblytheorytools.complexity_scores.compression_zlib_smi` and
friends) and {func}`~assemblytheorytools.complexity_scores.shannon_entropy` are
not published molecular complexity measures and need no citation.

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
  licence. ATT neither bundles nor redistributes the archive —
  {doc}`Protocol 3 <examples/protocol_3>` asks you to download it yourself from
  the DOI and point `CHEMOTION_IR_ARCHIVE` at the file.

## BibTeX

`att.bib` in the repository root holds an entry for every reference above, and
for the scientific Python stack ATT builds on — NumPy, SciPy, NetworkX, RDKit,
matplotlib and ASE. Please cite those too where your journal's conventions
allow it.
