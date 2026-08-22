# Strings

Assembly theory applies to any object built by joining parts, so a string is as
valid a target as a molecule. The elementary parts are characters, and a join
concatenates two pieces that have already been made.

## A single string

```python
import assemblytheorytools as att

ai, virt_obj, pathway = att.calculate_string_assembly_index("abracadabra")

print(ai)         # 7
print(virt_obj)   # ['a', 'b', 'c', 'r', 'd', 'ab', 'abr', 'abra', ...]
```

Building `abracadabra` character by character would take ten joins. Because
`abra` can be reused once it exists, the index is 7. As with molecules, the
order of `virt_obj` is not stable between runs.

The string calculator is a separate binary from the molecule one; it is bundled
with the package and resolved through `ASS_STR_PATH` (see
{doc}`../configuration`).

## Directed and undirected strings

`directed=True` (the default) treats the string as read in one direction only.
`directed=False` also allows a fragment to be reused reversed, which suits
sequences with no intrinsic reading direction:

```python
att.calculate_string_assembly_index("abracadabra", directed=False, mode="mol")
```

Undirected calculations only run through the molecule calculator, so pass
`mode="mol"` explicitly; otherwise the function switches to it and warns.

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

`mode="mol"` routes the joint calculation through the molecule calculator,
which is what supports sharing across components. {func}`~assemblytheorytools.tools_string.prep_joint_string_ai`
does the encoding, joining the inputs with a separator character that does not
appear in any of them:

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

Assembly index grows with sequence length, so start with short sequences and
raise `timeout` as needed.

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

print(observed, statistics.mean(baseline))   # 7 10
```

A random 11-character string over 26 letters has almost no repetition to
exploit, so it consistently scores 10. `abracadabra` scores 7 — the difference
is what its internal structure buys.

## See also

* {doc}`../api/assembly` — {func}`~assemblytheorytools.assembly.calculate_string_assembly_index` and {func}`~assemblytheorytools.assembly.calculate_string_assembly`.
* {doc}`../api/tools_string` — string preparation helpers.
* {doc}`pathways` — plotting a string pathway.
