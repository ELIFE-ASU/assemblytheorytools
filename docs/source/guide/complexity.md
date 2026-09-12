# Complexity scores

Assembly index is one complexity measure among many.
{mod}`assemblytheorytools.complexity_scores` implements the common alternatives
so they can be computed on the same molecules and compared directly.

Most molecular scores take an RDKit `Mol`:

```python
import assemblytheorytools as att

mol = att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")   # caffeine

att.bertz_complexity(mol)   # 924.42
att.wiener_index(mol)       # 1089
att.molecular_weight(mol)   # 194.19
att.count_bonds(mol)        # 25
att.count_non_h_bonds(mol)  # 15
```

## What is available

**Graph-theoretic indices** — computed from the molecular graph's topology:

| Score | Function | Defined in |
| --- | --- | --- |
| Bertz | {func}`~assemblytheorytools.complexity_scores.bertz_complexity` | [Bertz (1981)](https://doi.org/10.1021/ja00402a071) |
| Böttcher | {func}`~assemblytheorytools.complexity_scores.bottcher` | [Böttcher (2016)](https://doi.org/10.1021/acs.jcim.5b00723) |
| Wiener | {func}`~assemblytheorytools.complexity_scores.wiener_index` | [Wiener (1947)](https://doi.org/10.1021/ja01193a005) |
| Balaban | {func}`~assemblytheorytools.complexity_scores.balaban_index` | [Balaban (1982)](https://doi.org/10.1016/0009-2614(82)80009-2) |
| Randić | {func}`~assemblytheorytools.complexity_scores.randic_index` | [Randić (1975)](https://doi.org/10.1021/ja00856a001) |
| Kirchhoff | {func}`~assemblytheorytools.complexity_scores.kirchhoff_index` | [Klein & Randić (1993)](https://doi.org/10.1007/BF01164627) |

`wiener_index` returns an `int`; the rest return floats.

**Substructure and shape measures**:

* {func}`~assemblytheorytools.complexity_scores.spacial_score` — three-dimensional
  complexity; `normalise=True` divides by heavy atom count.
  [Krzyzanowski *et al.* (2023)](https://doi.org/10.1021/acs.jmedchem.3c00689)
* {func}`~assemblytheorytools.complexity_scores.proudfoot` — atom-environment
  complexity.
  [Proudfoot (2017)](https://doi.org/10.1016/j.bmcl.2017.03.008)
* {func}`~assemblytheorytools.complexity_scores.mc1` and
  {func}`~assemblytheorytools.complexity_scores.mc2` — molecular complexity
  measures. `mc2` returns an `int`: it is a count of atoms, not a score.
  [Buehler & Reymond (2025)](https://doi.org/10.1021/acs.jcim.5c00334)
* {func}`~assemblytheorytools.complexity_scores.shannon_entropy` — character
  entropy of an input string (for example, a SMILES string). This is an
  exception to the `Mol` input convention.

**Compression-based measures** — how far a standard compressor shrinks the
structure, a crude proxy for redundancy. Unlike the scores above these are not
published molecular complexity measures, so there is nothing to cite for them:

* {func}`~assemblytheorytools.complexity_scores.compression_zlib_smi`,
  {func}`~assemblytheorytools.complexity_scores.compression_bz2_smi`,
  {func}`~assemblytheorytools.complexity_scores.compression_lzma_smi` — take an
  RDKit `Mol`, standardise it internally, and compress the resulting SMILES.
* {func}`~assemblytheorytools.complexity_scores.compression_zlib_graph` and
  {func}`~assemblytheorytools.complexity_scores.compression_ratio_zlib_graph` —
  on the graph.

The SMILES compression helpers standardise the molecule internally, so pass the
`Mol` directly. The graph compression helpers instead take a NetworkX graph;
their serialised representation can reflect labels and insertion order, so
normalise inputs before comparing values from different sources.

**Bulk descriptors** —
{func}`~assemblytheorytools.complexity_scores.get_mol_descriptors` returns
RDKit's full descriptor set as a dictionary, which is the quickest way to get a
feature table for a modelling workflow.

## Similarity

Fingerprint similarity, for comparison against assembly-based similarity:

```python
glycine = att.smi_to_mol("NCC(=O)O")
alanine = att.smi_to_mol("CC(N)C(=O)O")

att.tanimoto_similarity(glycine, alanine)      # 0.327
att.dice_morgan_similarity(glycine, alanine)
```

Compare that against the assembly-theoretic score for the same pair:

```python
att.calculate_assembly_index_similarity(
    [att.smi_to_nx("NCC(=O)O"), att.smi_to_nx("CC(N)C(=O)O")],
    settings={"strip_hydrogen": True})   # 0.75
```

In a script this call needs an `if __name__ == "__main__":` guard, or
`parallel=False` — it runs its constituent calculations in a process pool. See
{doc}`parallel`.

The two disagree because they measure different things: Tanimoto compares
fingerprint bits, whereas the assembly score asks how much of the construction
work is shared.

## Comparing scores across a dataset

The usual workflow is to compute several scores over the same molecules and
correlate them:

```python
smiles = [...]
mols = [att.smi_to_mol(s) for s in smiles]
graphs = [att.smi_to_nx(s) for s in smiles]

ai, _, _ = att.calculate_assembly_index_parallel(graphs, dict(strip_hydrogen=True))
bertz = [att.bertz_complexity(m) for m in mols]

att.scatter_plot(bertz, ai, xlab="Bertz", ylab="Assembly index")
```

{func}`~assemblytheorytools.tools_data.get_r`,
{func}`~assemblytheorytools.tools_data.get_r2` and
{func}`~assemblytheorytools.tools_data.get_rmsd` give the correlation
statistics, and
{func}`~assemblytheorytools.tools_plotting.plot_heatmap` handles the density
plot when the dataset is too large for a scatter.

`examples/advanced/kegg_compounds/kegg_c_complexity_matrix.py` does exactly this
across seven scores on KEGG compounds.

## See also

* {doc}`../citing` — the full reference for every score above.
* {doc}`../api/complexity_scores` — every scoring function.
* {doc}`../api/tools_data` — sampling, filtering and fitting helpers.
* {doc}`../examples/protocol_2` — assembly index against molecular weight at scale, for CBRDB, PubChem and your own molecules.
