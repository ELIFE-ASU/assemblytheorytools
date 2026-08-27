# Concepts

This page explains the vocabulary used throughout the API. It is deliberately
short on theory: {doc}`theory` covers the background, {doc}`glossary` gives the
formal definitions, and the papers listed under {doc}`citing` are the reference.

## The assembly index

The **assembly index** of an object is the smallest number of joining steps
needed to build it from elementary parts, where every intermediate that has
already been constructed may be reused for free.

The reuse rule is what makes the measure interesting. Building `abracadabra`
character by character takes ten joins, but once `abra` exists it can be joined
to itself and to the remaining fragment, and the index drops to 7. The index
therefore rewards *internal repetition*, not merely size.

For molecules the elementary parts are bonds, so a molecular assembly index
counts bond-forming steps over the molecular graph.

## Virtual objects

A **virtual object** (VO) is an intermediate that appears on the assembly path.
Every VO is either an elementary part or the result of joining two earlier VOs,
and the final object is the last VO in the sequence.

`calculate_assembly_index` returns the virtual objects as graphs. Convert them
back to SMILES with {func}`~assemblytheorytools.tools_graph.nx_to_smi`:

```python
virt_smiles = [att.nx_to_smi(g, add_hydrogens=False) for g in virt_obj]
```

The order of the returned list is not stable between runs, so compare virtual
objects as a set rather than by position.

## The assembly pathway

The **pathway** is the record of how the object was built: a
{class}`~networkx.DiGraph` whose nodes are virtual objects and whose edges are
joining operations pointing from inputs to output. Elementary parts have
in-degree zero; the target object has out-degree zero.

An object usually has more than one shortest pathway. The calculator returns
one of them; {func}`~assemblytheorytools.find_other_paths.all_shortest_paths`
enumerates the alternatives.

See {doc}`guide/pathways` for parsing, levelling and plotting pathways.

## Joint assembly

The **joint assembly index** of several objects is the index of building all of
them together, sharing intermediates across the set. It is computed by placing
the objects in one disconnected graph — for molecules, a SMILES string with
`.` separators:

```python
joint = att.smi_to_nx("NCC(=O)O.CC(N)C(=O)O")   # glycine and alanine
ai, virt_obj, pathway = att.calculate_assembly_index(joint, strip_hydrogen=True)
```

Because the two amino acids share most of their structure, the joint index (4)
is far below the sum of the separate indices (3 + 4 = 7). The gap is a measure
of how much the objects have in common, and is what
{func}`~assemblytheorytools.assembly.calculate_assembly_index_similarity` turns
into a similarity score.

By default `calculate_assembly_index` applies a correction for the number of
disconnected components (`joint_corr=True`). Set it to `False` to get the raw
value returned by the calculator.

## Objects ATT can handle

| Object | Built with | Guide |
| --- | --- | --- |
| Molecules | {func}`~assemblytheorytools.tools_graph.smi_to_nx`, RDKit `Mol` | {doc}`guide/molecules` |
| Strings, directed and undirected | {func}`~assemblytheorytools.assembly.calculate_string_assembly_index` | {doc}`guide/strings` |
| Arbitrary labelled graphs | NetworkX `Graph` with `color` attributes | {doc}`guide/graphs` |
| Crystal structures | {func}`~assemblytheorytools.tools_cell.cif_to_nx` | {doc}`guide/graphs` |

## Calculator backends

ATT does not implement the search itself; it prepares input, drives an external
calculator and parses the result. Three backends are available.

assemblyCPP
: The default. A C++ branch-and-bound calculator, invoked by
  {func}`~assemblytheorytools.assembly.calculate_assembly_index`. Precompiled
  static binaries ship in the wheel, and `ASS_PATH` overrides them. This is the
  only backend that returns virtual objects and a pathway.

assembly-theory (Rust)
: Reached through
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_rust`, which
  returns the index alone. It always strips hydrogens, so compare it against
  `strip_hydrogen=True` results.
  {func}`~assemblytheorytools.assembly.calculate_assembly_depth_rust` gives the
  molecule's minimum assembly depth, and
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_rust_search`
  exposes the search options and reports how many duplicate subgraph pairs and
  assembly states the search saw. That function also reconstructs minimum
  assembly pathways, on releases that support it — see
  [Pathways](guide/pathways.md).

assemblycfg
: A fast approximate method based on context-free grammars, used by the
  upper- and lower-bound helpers
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_upper_bound`
  and
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_lower_bound`.
  Use it when an exact search would be too slow.

## Hydrogens

Whether hydrogens are part of the graph changes the answer, and the backends
disagree by default:

```python
graph = att.smi_to_nx("CCO")

att.calculate_assembly_index(graph)[0]                      # 6
att.calculate_assembly_index(graph, strip_hydrogen=True)[0]  # 1
att.calculate_assembly_index_rust(graph)                     # 1
```

Most published molecular assembly indices are hydrogen-stripped. Pass
`strip_hydrogen=True` unless you specifically want hydrogens counted, and never
compare a stripped index against an unstripped one.

`strip_hydrogen=True` strips a copy, so the graph you passed keeps its
hydrogens and stays reusable — as above, where one graph serves all three
calls.

## Complexity scores

Assembly index is one complexity measure among many, and
{mod}`assemblytheorytools.complexity_scores` implements the usual alternatives —
Bertz, Böttcher, Wiener, Balaban, Randić, Kirchhoff, spacial score, Proudfoot,
MC1/MC2, Shannon entropy and several compression-based measures — so they can be
compared on the same molecules. See {doc}`guide/complexity`.
