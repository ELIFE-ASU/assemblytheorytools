# Molecules

## From SMILES to an assembly index

{func}`~assemblytheorytools.tools_graph.smi_to_nx` turns a SMILES string into a
NetworkX graph with the node and edge attributes the calculator expects, and
{func}`~assemblytheorytools.assembly.calculate_assembly_index` runs the
calculation:

```python
import assemblytheorytools as att

graph = att.smi_to_nx("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")   # caffeine
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

print(ai)              # 9
print(len(virt_obj))   # 14
```

Pass `strip_hydrogen=True` for anything you intend to compare against published
molecular assembly indices. Without it the hydrogens are counted as part of the
structure and the index is much larger — see
[Hydrogens](../concepts.md#hydrogens). Stripping works on a copy, so the graph
you pass keeps its hydrogens and can be reused for further calculations.

RDKit molecules work directly, so there is no need to round-trip through SMILES
if you already have a `Mol`:

```python
mol = att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)
```

Other entry points into the same graph format:

| From | Function |
| --- | --- |
| SMILES | {func}`~assemblytheorytools.tools_graph.smi_to_nx` |
| InChI | {func}`~assemblytheorytools.tools_graph.inchi_to_nx` |
| RDKit `Mol` | {func}`~assemblytheorytools.tools_graph.mol_to_nx` |
| ASE `Atoms` | {func}`~assemblytheorytools.tools_atoms.atoms_to_nx` |
| CIF file | {func}`~assemblytheorytools.tools_cell.cif_to_nx` |
| PubChem name or CID | {func}`~assemblytheorytools.tools_data.pubchem_name_to_nx`, {func}`~assemblytheorytools.tools_data.pubchem_id_to_nx` |

## Reading the virtual objects

`virt_obj` comes back as graphs. Convert them to SMILES to inspect them:

```python
virt_smiles = [att.nx_to_smi(g, add_hydrogens=False) for g in virt_obj]
```

For caffeine this yields fragments such as `C=O`, `CN`, `CC1=CN=CN1C` and the
full molecule. The list order is not stable between runs — treat it as a set.

## Joint assembly

Put several molecules in one disconnected graph to compute their joint assembly
index, which shares intermediates across the set. In SMILES, `.` separates
components:

```python
glycine, alanine = "NCC(=O)O", "CC(N)C(=O)O"

separate = [att.calculate_assembly_index(att.smi_to_nx(s), strip_hydrogen=True)[0]
            for s in (glycine, alanine)]
joint = att.calculate_assembly_index(
    att.smi_to_nx(f"{glycine}.{alanine}"), strip_hydrogen=True)[0]

print(separate, sum(separate), joint)   # [3, 4] 7 4
```

The joint index is 4 against a separate total of 7: the two amino acids share
most of their structure, so nearly everything built for one is reused for the
other.

{func}`~assemblytheorytools.assembly.calculate_assembly_index_similarity` turns
that gap into a score between 0 and 1:

```python
att.calculate_assembly_index_similarity(
    [att.smi_to_nx(glycine), att.smi_to_nx(alanine)])   # 0.636...
```

Related helpers:
{func}`~assemblytheorytools.assembly.calculate_sum_assembly_index` for the
separate total, {func}`~assemblytheorytools.assembly.calculate_assembly_index_ratio`
and {func}`~assemblytheorytools.assembly.calculate_assembly_index_pairwise_joint`
for pairwise comparisons across a set.

## Choosing a backend

The default backend returns the index, the virtual objects and the pathway. If
you only need the number, the Rust backend is faster:

```python
att.calculate_assembly_index_rust(graph)    # 9 for caffeine
```

It returns a bare integer and always strips hydrogens, so only compare it with
`strip_hydrogen=True` results.

When an exact search is too slow, bound the answer instead:

```python
att.calculate_assembly_index_upper_bound(graph)
att.calculate_assembly_index_lower_bound(graph)
```

These use the approximate `assemblycfg` method and return quickly on molecules
where the exact search would time out.

## When a calculation times out

The search is exponential in the worst case. `calculate_assembly_index` gives
the external calculator `timeout=100.0` seconds and abandons the run rather than
returning a partial answer. For large molecules, either raise the timeout or
switch to the bound helpers above.

To see what the calculator actually did, keep the working directory:

```python
ai, virt_obj, pathway = att.calculate_assembly_index(
    graph, strip_hydrogen=True, debug=True, save_dir=True)
```

This prints the calculator's output and leaves the generated input file and raw
log in place for inspection.

## See also

* {doc}`../api/assembly` — every assembly index entry point.
* {doc}`../api/tools_graph` — graph conversion and manipulation.
* {doc}`../api/tools_mol` — RDKit-level molecule helpers.
* {doc}`parallel` — running many molecules at once.
* {doc}`pathways` — inspecting and plotting the result.
