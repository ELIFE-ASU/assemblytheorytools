```{include} ../../../examples/protocols/2/README.md
:parser: myst
```

## Results

```{figure} ../../../examples/protocols/2/assembly_index_heatmap.png
:alt: Density heatmap of assembly index against molecular weight
:width: 100%

Assembly index against molecular weight across 10,000 CBRDB molecules. The
density plot is used in place of a scatter because the point count makes
individual markers unreadable.
```

```{figure} ../../../examples/protocols/2/molecule_grid.png
:alt: Grid of molecular structures sorted by assembly index
:width: 100%

A subset of the analysed molecules with their nicknames and assembly indices,
sorted by index.
```

## Running it

```bash
cd examples/protocols/2
python protocol_2.py
```

This is the longest-running protocol: it calculates assembly indices for 10,000
molecules. It sets a per-molecule timeout and strips hydrogens to keep the load
manageable, then filters to molecules with an index of 1 or greater — an index
below that marks a calculation that did not produce a usable answer.

Tune the worker count to your machine before running; see
[Parallel calculations](../guide/parallel.md).

## Related documentation

* [Parallel calculations](../guide/parallel.md) — `mp_calc` and
  `calculate_assembly_index_parallel`.
* [Complexity scores](../guide/complexity.md) — comparing assembly index with
  other measures.
