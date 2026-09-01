```{include} ../../../examples/basic/README.md
:parser: myst
```

## Running them

```bash
cd examples/basic
python 1_simple_molecule_example.py
```

The scripts are numbered in increasing order of complexity, so working through
them in order is the fastest way to get oriented.

Scripts 4 and 7 call `att.plot_digraph_metro`, which is Linux-only; both guard
that call so the rest of the script still runs elsewhere.

## Related documentation

* [Molecules guide](../guide/molecules.md) — scripts 1, 4 and 6.
* [Strings guide](../guide/strings.md) — scripts 2 and 5.
* [Arbitrary graphs guide](../guide/graphs.md) — script 3.
* [Pathways guide](../guide/pathways.md) — script 7.
* [Spectroscopy and mass spectrometry](../guide/mass_spectrometry.md) — script 8.
* [Molecules guide: Rust backend](../guide/molecules.md#choosing-a-backend) — script 9.
