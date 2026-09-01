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

For the NetworkX input in the first example, `virt_obj` comes back as graphs.
Convert them to SMILES to inspect them:

```python
virt_smiles = [att.nx_to_smi(g, add_hydrogens=False) for g in virt_obj]
```

For caffeine this yields fragments such as `C=O`, `CN`, `CC1=CN=CN1C` and the
full molecule. With an RDKit `Mol` input, ATT performs this conversion before
returning, so `virt_obj` is already a list of SMILES strings. The list order is
not stable between runs — treat it as a set.

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

{func}`~assemblytheorytools.assembly.calculate_assembly_index_similarity`
reports `(sum of separate indices / joint index) - 1`. For two objects the
score lies between 0 and 1; with more inputs it can be larger:

```python
att.calculate_assembly_index_similarity(
    [att.smi_to_nx(glycine), att.smi_to_nx(alanine)],
    settings={"strip_hydrogen": True})   # 0.75
```

The helper enables exact mode by default so that a timed-out upper bound is not
mistaken for a similarity value.

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

Two further functions reach the same backend.
{func}`~assemblytheorytools.assembly.calculate_assembly_depth_rust` gives the
molecule's minimum assembly depth, and
{func}`~assemblytheorytools.assembly.calculate_assembly_index_rust_search`
exposes the search itself:

```python
att.calculate_assembly_depth_rust(att.smi_to_nx("c1ccccc1"))   # 3 for benzene

result = att.calculate_assembly_index_rust_search(graph, timeout=10)
result.index, result.num_matches, result.states_searched
```

:::{warning}
The depth search is far more expensive than the index search and takes no
timeout, so it is only practical on small molecules. Benzene returns instantly
and naphthalene takes around four minutes, while both of their indices come back
in well under a second.
:::

`num_matches` counts the edge-disjoint isomorphic subgraph pairs the search had
to consider and `states_searched` is `None` if the timeout fired, which together
say whether a slow molecule is slow because it is large or because it is
repetitive. The search options — `canonize`, `parallel`, `memoize`, `kernel` and
`bounds` — are documented on the function; the defaults are the backend's own.
On releases that support it, `max_pathways` also reconstructs the minimum
assembly pathways; see [Pathways](pathways.md).

{func}`~assemblytheorytools.assembly.get_molecule_info_rust` returns a DOT
string describing the graph the backend actually builds, which is the quickest
way to confirm that hydrogens were dropped or that a ring was kekulised.

When an exact search is too slow, bound the answer analytically instead:

```python
att.calculate_assembly_index_upper_bound(graph)
att.calculate_assembly_index_lower_bound(graph)
```

The upper bound is the number of bonds minus one. The lower bound is the
addition-chain length for fewer than 1,000 bonds, falling back to `log2` from
1,000 onward. Pass `strip_hydrogen=True` when the comparison should use only
heavy-atom bonds.

## When a calculation times out

The search is exponential in the worst case. `calculate_assembly_index` gives
the external calculator `timeout=100.0` seconds. If that limit is reached, the
default mode returns the best upper bound recorded so far, or `-1` if none was
found; virtual objects and a pathway may be unavailable. Pass `exact=True` to
return `-1` instead of an upper bound, raise the timeout, or use the bound
helpers above.

To see what the calculator actually did, keep the working directory:

```python
ai, virt_obj, pathway = att.calculate_assembly_index(
    graph, strip_hydrogen=True, debug=True, save_dir=True)
```

This prints ATT's diagnostics and leaves the generated input file and
`assembly_output.log` in place. The calculator's own standard output and error
are redirected into that log rather than echoed to the terminal.

## See also

* {doc}`../api/assembly` — every assembly index entry point.
* {doc}`../api/tools_graph` — graph conversion and manipulation.
* {doc}`../api/tools_mol` — RDKit-level molecule helpers.
* {doc}`parallel` — running many molecules at once.
* {doc}`pathways` — inspecting and plotting the result.
