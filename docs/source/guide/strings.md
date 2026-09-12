# Strings

Assembly theory applies to any object built by joining parts, so a string is as
valid a target as a molecule. The elementary parts are characters, and a join
concatenates two pieces that have already been made.

## A single string

```python
import assemblytheorytools as att

ai, virt_obj, pathway = att.calculate_string_assembly_index("abracadabra")

print(ai)         # 7
print(virt_obj)   # ['a', 'b', 'r', 'c', 'd', 'ab', 'abr', 'abra', ...]
```

Building `abracadabra` character by character would take ten joins. Because
`abra` can be reused once it exists, the index is 7. As with molecules, the
order of `virt_obj` is not stable between runs.

String and molecule calculations use the same `ParallelAssemblyCpp` executable,
resolved through `ASS_PATH`. Set `ASS_STR_PATH` only to point string
calculations at a different build (see {doc}`../configuration`).

## Directed and undirected strings

`directed=True` (the default) treats the string as read in one direction only.
`directed=False` also allows a fragment to be reused reversed, which suits
sequences with no intrinsic reading direction:

```python
att.calculate_string_assembly_index("abracadabra", directed=False, mode="mol")
```

Undirected calculations only run through the molecule calculator, so pass
`mode="mol"` explicitly; otherwise the function switches to it and warns.

## An approximate mode for long strings

A third mode, `mode="cfg"`, skips the external calculator entirely and returns a
RePair smallest-grammar **upper bound** on the index, together with its pathway.
It never returns a value below the true index, which is what makes it safe for
screening, and it is the practical choice for sequences where the exact search
does not finish. It takes no `cpp_options`.

```python
att.calculate_string_assembly_index("abracadabra", mode="cfg")[0]   # 7
```

## Plotting a string pathway

{func}`~assemblytheorytools.tools_plotting.plot_pathway` draws string pathways
with `plot_type="string"`, but it reads each node's `vo` attribute and a string
pathway's nodes carry no attributes — the node *is* its own label. Set them
first:

```python
import networkx as nx

ai, virt_obj, pathway = att.calculate_string_assembly_index("abracadabra")
nx.set_node_attributes(pathway, {n: n for n in pathway}, "vo")

fig, ax = att.plot_pathway(pathway, plot_type="string")
```

Without that line the call raises `KeyError: 'vo'`.

## Joint assembly across several strings

Pass a list to compute a joint index, sharing intermediates across the set:

```python
ai, virt_obj, pathway = att.calculate_string_assembly_index(
    ["abracadabra", "abra"], directed=False, mode="mol")

print(ai)   # 7
```

Adding `abra` to `abracadabra` costs nothing: `abra` is already built as an
intermediate of the longer string, so the joint index equals the index of
`abracadabra` alone.

List inputs share intermediates in both the default directed string mode and
the molecule-graph mode. This example uses `mode="mol"` because undirected
strings require it. {func}`~assemblytheorytools.tools_string.prep_joint_string_ai`
does the multi-string encoding, joining the inputs with a separator character
that does not appear in any of them:

```python
joined, separators = att.prep_joint_string_ai(["abracadabra", "abra"])
print(joined, separators)   # abracadabra0abra ['0']
```

## Sequence data

{func}`~assemblytheorytools.tools_string.load_fasta` reads a FASTA file into a
single string, which makes protein and nucleotide sequences directly usable:

```python
sequence = att.load_fasta("protein.fasta")
ai, virt_obj, pathway = att.calculate_string_assembly_index(sequence)
```

Assembly index grows with sequence length, so start with short sequences, raise
`timeout` as needed, and switch to `mode="cfg"` when the exact search stops
finishing.

Other helpers in {mod}`assemblytheorytools.tools_string`:

* {func}`~assemblytheorytools.tools_string.get_unique_char` — returns a
  character *not* present in the input, which is how `prep_joint_string_ai`
  picks a safe separator.
* {func}`~assemblytheorytools.tools_string.generate_random_strings` — random
  lowercase strings, for null models and benchmarking.
* {func}`~assemblytheorytools.tools_string.get_dir_str_molecule` and
  {func}`~assemblytheorytools.tools_string.get_undir_str_molecule` — the graph
  encoding of a string, if you want to inspect what the calculator receives.

## Comparing against a random baseline

An index on its own is hard to interpret; what matters is how it compares with
unstructured strings of the same length.
{func}`~assemblytheorytools.tools_string.generate_random_strings` takes the
number of strings and their length, and draws from lowercase letters:

```python
import statistics

sequence = "abracadabra"
observed = att.calculate_string_assembly_index(sequence)[0]

baseline = [att.calculate_string_assembly_index(s)[0]
            for s in att.generate_random_strings(20, len(sequence))]

print(observed, statistics.mean(baseline))   # 7; baseline varies
```

A random 11-character string over 26 letters has almost no repetition to
exploit, so its mean is typically close to the character-by-character upper
bound of 10. The sample is unseeded and varies between runs. `abracadabra`
scores 7 — the difference is what its internal structure buys.

## See also

* {doc}`../api/assembly` — {func}`~assemblytheorytools.assembly.calculate_string_assembly_index` and {func}`~assemblytheorytools.assembly.calculate_string_assembly`.
* {doc}`../api/tools_string` — string preparation helpers.
* {doc}`pathways` — levelling and plotting. Draw a string pathway with
  {func}`~assemblytheorytools.tools_plotting.plot_pathway` and
  `plot_type="string"`, after labelling the nodes (see below).
