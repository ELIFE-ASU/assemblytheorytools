# Reassembly

{mod}`assemblytheorytools.reassembler` builds molecules by applying SMARTS
reaction templates, following
[Liu *et al.* (2021)](https://doi.org/10.1126/sciadv.abj2465) for the
fragment-pool construction and
[Pagel *et al.* (2026)](https://doi.org/10.1021/acs.jcim.6c00939) for the
generative search. Where {doc}`enumeration` performs graph joins by vertex
identification and edge partitioning, reassembly works through chemical
templates, so the structures it produces are plausible reaction products.

## Joining two molecules

{func}`~assemblytheorytools.reassembler.assemble` takes two molecules and the
number of sites to join, and returns the product:

```python
from rdkit import Chem
import assemblytheorytools as att

product = att.assemble(Chem.MolFromSmiles("OCC(O)CO"),
                       Chem.MolFromSmiles("C=C"), 1)
print(Chem.MolToSmiles(product))   # C=C(O)C(O)CO
```

It returns `None` when no template applies to the pair.

## Cyclisation

{func}`~assemblytheorytools.reassembler.origami` applies the intramolecular
templates, folding a molecule onto itself to give every ring-closed product:

```python
products = att.origami(Chem.MolFromSmiles("OCC(O)CO"))
print(sorted(Chem.MolToSmiles(m) for m in products))
# ['OC1COC1', 'OCC1CO1']
```

Glycerol closes into an epoxide and an oxetane. The order of the returned list is
not stable between runs, so sort or compare as a set. Applying `origami` to a
product that has no further ring to close returns that molecule unchanged and
prints `No origami products found...`.

The template libraries themselves are available as `origami_smarts()` and
`assemble_smarts()`. Both need a submodule import — they are not re-exported at
the package root:

```python
from assemblytheorytools.reassembler import assemble_smarts, origami_smarts
```

## Molecules and molecule spaces

For anything beyond a single step, the class API drives the search.
{class}`~assemblytheorytools.reassembler.Molecule` wraps a SMILES string
together with its assembly pathway:

```python
molecules = []
for smiles in ["CCO", "CC=O"]:            # ethanol, acetaldehyde
    molecule = att.Molecule(smiles=smiles)
    molecule.reconstruct_pathway()
    molecule.construct_layered_graph()
    molecules.append(molecule)
```

{class}`~assemblytheorytools.reassembler.MoleculeSpace` holds a set of them and
merges their pathways into one graph — the estimated joint assembly, in which
an intermediate shared by two molecules appears once:

```python
space = att.MoleculeSpace(molecules=molecules)
space.construct_joined_graph()

graph = space.joined_assembly_graph
print(sorted(graph.nodes))
# ['C=O', 'CC', 'CC=O', 'CCO', 'CO']
```

Ethanol and acetaldehyde share the `CC` fragment, so the joined graph has five
nodes rather than the six the two pathways hold separately.

{class}`~assemblytheorytools.reassembler.MoleculeGenerationAssemblyPool` is the
driver: it holds the fragments of a
{class}`~assemblytheorytools.reassembler.MoleculeSpace` grouped by assembly
level and runs the generation-by-generation search.
{class}`~assemblytheorytools.reassembler.Assemble` is the stateless bond-forming
engine it calls. A full run builds the pool from a `MoleculeSpace`; the single
join below passes a bare graph of starting fragments instead, which is all
`combine_fragments_layer` needs:

```python
import random
import networkx as nx

random.seed(10)

pool_graph = nx.DiGraph()
pool_graph.add_nodes_from(["NCC(O)=O", "CC(N)C(O)=O"])   # glycine, alanine
generation = att.MoleculeGenerationAssemblyPool(pool_graph)

product = generation.combine_fragments_layer(
    fragment1="NCC(O)=O", fragment2="CC(N)C(O)=O", assemble_object=att.Assemble())
print(product)   # CC(N)C(=O)OC(=O)CN
```

This path uses no SMARTS at all.
{class}`~assemblytheorytools.reassembler.Assemble` fuses the two fragments by
*overlapping* atoms, drawing how many atoms to overlap from empirical weights,
so the same pair can join at different atoms on different runs. Seed
{mod}`random` when you need a reproducible product, as above.

## Constraining the search

The space grows combinatorially with each generation, so filter aggressively:

* {func}`~assemblytheorytools.reassembler.get_unique_mols` — deduplicate a list
  of products.
* {func}`~assemblytheorytools.reassembler.degree_unsaturation` and
  {func}`~assemblytheorytools.reassembler.get_num_atom` — cheap structural
  bounds for rejecting candidates early.
* `filter_mol`, `conformation_filter` and `valence_check` — chemical
  plausibility checks (submodule import).

`examples/advanced/reassemble/` contains two worked scripts: `reassemble_aa.py`
reassembles amino acids, and `reassemble_vo_space.py` explores the space spanned
by a set of virtual objects.

## Parsing a reassembly log

{class}`~assemblytheorytools.reassembler.ParsePathwayLog` parses the pathway log
the *assembly-index* calculator writes — the same log
`Molecule.reconstruct_pathway` captures — into a layered
{class}`~networkx.MultiDiGraph`. It takes the log contents, not a filename:

```python
from assemblytheorytools.reassembler import ParsePathwayLog
```

## Relation to the assembly index

Reassembly and the assembly index answer different questions. The index asks for
the *shortest* construction of one known target. Reassembly asks what is
*reachable* from a starting pool — it enumerates a space rather than optimising
within one. The two combine naturally: take the virtual objects from an assembly
pathway as the starting pool, and reassembly shows what else those same building
blocks could have produced.

## See also

* {doc}`../api/reassembler` — the full class and function reference.
* {doc}`enumeration` — graph-level neighbourhood exploration.
* {doc}`../api/tools_mol` — fragment standardisation and valence helpers.
