```{include} ../../../examples/protocols/4/README.md
:parser: myst
```

## Results

```{figure} ../../../examples/protocols/4/processed_MS2.png
:alt: Processed MS2 spectrum with fragmentation tree
:width: 100%

The processed MS2 spectrum with its fragmentation tree overlaid — the fragments
shown are exactly those that feed the recursive estimate.
```

```{figure} ../../../examples/protocols/4/example_atoms.png
:alt: 3D atomic structure of the test compound
:width: 60%
:align: center

The 3D structure of the test compound. It is shown for reference only; the
estimate uses nothing but the spectra.
```

## Running it

```bash
cd examples/protocols/4
python protocol_4.py
```

The sample data is bundled as `Sample_#15_Stepped_MS3.tar.xz` and unpacked by
the script, so no external download is needed.

## Related documentation

* [Spectroscopy and mass spectrometry](../guide/mass_spectrometry.md) — the
  fragmentation-tree route in detail.
* [API: recursive_ma](../api/recursive_ma.rst) — the MA estimator.
* [API: tools_mzml](../api/tools_mzml.rst) — mzML parsing.
