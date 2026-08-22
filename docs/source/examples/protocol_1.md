```{include} ../../../examples/protocols/1/README.md
:parser: myst
```

## Results

```{figure} ../../../examples/protocols/1/mol_pathway_example.png
:alt: Molecular assembly pathway for the combined system
:width: 100%

The joint assembly pathway for the combined molecular system, showing the
virtual objects shared between the inputs.
```

```{figure} ../../../examples/protocols/1/str_pathway_example.png
:alt: String assembly pathway
:width: 100%

The assembly pathway for the string `gggfhhhvg`, with the reused substrings
appearing as virtual objects.
```

## Running it

```bash
cd examples/protocols/1
python protocol_1.py
```

## Related documentation

* [Molecules guide](../guide/molecules.md) — single and joint molecular assembly.
* [Strings guide](../guide/strings.md) — string assembly indices.
* [Pathways guide](../guide/pathways.md) — plotting and inspecting the result.
