```{include} ../../../examples/protocols/3/README.md
:parser: myst
```

## Results

```{figure} ../../../examples/protocols/3/example_ir_spectrum.svg
:alt: IR spectrum with identified peaks
:width: 100%

The IR spectrum of the example molecule with its identified peaks. Peak finding
runs over the 400–1500 cm⁻¹ fingerprint region after a Savitzky-Golay smooth.
```

```{figure} ../../../examples/protocols/3/example_atoms.png
:alt: 3D atomic structure of the example molecule
:width: 60%
:align: center

The 3D structure of the same molecule, rendered with `att.plot_ase_atoms`.
```

```{figure} ../../../examples/protocols/3/ir_ai_correlation_heatmap.svg
:alt: Observed against predicted assembly index
:width: 100%

Observed against predicted assembly index across the dataset. The predicted
value comes from a linear model on IR peak count alone.
```

## Running it

```bash
cd examples/protocols/3
python protocol_3.py /path/to/10.22000-OGoEQGlsZGElrgst.tar
```

The Chemotion IR archive is external data and is not included in the
repository. Pass the downloaded `.tar` file as the positional argument.

## Related documentation

* [Spectroscopy and mass spectrometry](../guide/mass_spectrometry.md) — the IR
  route in detail.
* [Complexity scores](../guide/complexity.md) — correlation statistics.
