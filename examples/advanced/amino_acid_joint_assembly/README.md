# Amino acid joint assembly

This example illustrates the joint assembly of amino acids.
It showcases how to combine multiple amino acid residues into a single disjointed graph and determines the
co-construction — the shared assembly pathway that builds all of them together, reusing every intermediate across the
set.

`amino_acid_joint_assembly.py` works with glycine, alanine and serine. Because the three share most of their structure,
the joint assembly index sits well below the sum of their individual indices; the size of that gap is the quantity of
interest.

The script raises `timeout` to 600 seconds. Joint calculations are markedly more expensive than single-molecule ones —
the search space covers every way of sharing intermediates between components — so the default 100 second limit is not
enough here.

The example could be extended to include more amino acids or different types of residues, but note that cost grows
sharply with the number of components.

See the [joint assembly section](https://assemblytheorytools.readthedocs.io/en/latest/guide/molecules.html#joint-assembly)
of the molecules guide.
