# Route map

One table per family of quantities, listing what each one needs, what it
returns, and what it is for. Use it to find the right function; use
{doc}`concepts` for what the terms mean, {doc}`glossary` for their formal
definitions, and the {doc}`guide/index` for worked walkthroughs.

Everything named here is re-exported at the package root, so
`att.calculate_assembly_index` and
`assemblytheorytools.assembly.calculate_assembly_index` are the same function.

## Start from what you have

| You have | Start with | Guide |
| --- | --- | --- |
| A SMILES string or an RDKit `Mol` | {func}`~assemblytheorytools.tools_graph.smi_to_nx`, then {func}`~assemblytheorytools.assembly.calculate_assembly_index` | {doc}`guide/molecules` |
| A sequence or any text string | {func}`~assemblytheorytools.assembly.calculate_string_assembly_index` | {doc}`guide/strings` |
| A labelled NetworkX graph | {func}`~assemblytheorytools.assembly.calculate_assembly_index` | {doc}`guide/graphs` |
| A crystal structure in a CIF file | {func}`~assemblytheorytools.tools_cell.cif_to_nx`, then the same calculation | {doc}`guide/graphs` |
| Several objects to treat as one | {func}`~assemblytheorytools.tools_graph.join_graphs`, then the same calculation | {doc}`guide/molecules` |
| Thousands of structures | {func}`~assemblytheorytools.assembly.calculate_assembly_index_parallel` | {doc}`guide/parallel` |
| A tandem-MS fragmentation tree | {class}`~assemblytheorytools.recursive_ma.MAEstimator` | {doc}`guide/mass_spectrometry` |
| An infrared spectrum | {func}`~assemblytheorytools.tools_data.estimate_ai_from_ir_peaks` | {doc}`guide/mass_spectrometry` |
| A pathway from an earlier run | {func}`~assemblytheorytools.construction.parse_pathway_file` | {doc}`guide/pathways` |

## Core assembly quantities

| Quantity | Function | Input | Output | Use it for |
| --- | --- | --- | --- | --- |
| {term}`Assembly index` $a$ | {func}`~assemblytheorytools.assembly.calculate_assembly_index` | NetworkX `Graph` or RDKit `Mol` | `(index, virtual objects, pathway)` | The default exact calculation for molecules, graphs and crystal cells |
| Assembly index without the pathway | {func}`~assemblytheorytools.assembly.calculate_assembly_index_rust` | NetworkX `Graph` or RDKit `Mol` | `int` | Faster molecular indices when the pathway is not needed; always strips hydrogens |
| Assembly index with search detail | {func}`~assemblytheorytools.assembly.calculate_assembly_index_rust_search` | Molecule plus search options | {class}`~assemblytheorytools.assembly.RustSearchResult`: index, duplicate-subgraph matches, states searched, pathways | Tuning the search, and reconstructing several minimum pathways |
| String assembly index | {func}`~assemblytheorytools.assembly.calculate_string_assembly_index` | `str` or `list[str]` | `(index, virtual objects, pathway)` | Sequences and text; `directed=False` for undirected strings, `mode="cfg"` for a fast approximation |
| {term}`Virtual objects` $o_v$ | Second return value of the calculations above | — | Graphs for a NetworkX input, SMILES for an RDKit `Mol` | Reading off the reusable intermediates; the collection is unordered |
| {term}`Path` | Third return value of the calculations above | — | {class}`~networkx.DiGraph`, nodes are virtual objects and edges are joins | Plotting with {func}`~assemblytheorytools.tools_plotting.plot_pathway`, levelling, comparing routes |
| Minimum {term}`assembly depth` $d$ | {func}`~assemblytheorytools.assembly.calculate_assembly_depth_rust` | NetworkX `Graph` or RDKit `Mol` | `int` | The object's minimum depth under concurrent joins; substantially more expensive than the index, with no timeout |
| Assembly depth of one pathway | {func}`~assemblytheorytools.construction.assign_levels` | Topologically ordered pathway | The same graph, each node annotated with `level` | Layer-by-layer inspection and layout; this is the depth of *that* pathway, not the object's minimum |
| Joining-operation index | {func}`~assemblytheorytools.assembly.calculate_assembly_index_jo` | NetworkX `Graph` or RDKit `Mol` | `(jo, virtual objects, pathway)` | Counting joining operations rather than steps along the shortest path |

## Ensembles and comparisons

| Quantity | Function | Input | Output | Use it for |
| --- | --- | --- | --- | --- |
| {term}`Assembly` $A$ | {func}`~assemblytheorytools.assembly.calculate_assembly` | List of graphs and their {term}`copy numbers <Copy number>` `n_i` | `float` | Quantifying the selection implied by a measured ensemble |
| Assembly $A$ for strings | {func}`~assemblytheorytools.assembly.calculate_string_assembly` | List of strings and their copy numbers | `float` | The same quantity for sequence data |
| Joint assembly index | {func}`~assemblytheorytools.assembly.calculate_assembly_index` on a disconnected graph | Graphs merged with {func}`~assemblytheorytools.tools_graph.join_graphs`, or a `.`-separated SMILES | `(index, virtual objects, pathway)` | The cost of building a whole set together, sharing intermediates |
| Sum of separate indices | {func}`~assemblytheorytools.assembly.calculate_sum_assembly_index` | List of graphs | `int`, or `-1` if any calculation failed | The no-sharing baseline that a joint index is judged against |
| Shared-assembly score | {func}`~assemblytheorytools.assembly.calculate_assembly_index_similarity` | List of graphs | `float`, $\mathrm{sum}/\mathrm{joint} - 1$; between 0 and 1 for a pair | How much construction work a set of objects shares |
| Semi-metric distance | {func}`~assemblytheorytools.assembly.calculate_assembly_index_semi_metric` | Two graphs of the same type | `float`, $2 \times \mathrm{joint} - \mathrm{sum}$; `0.0` for isomorphic inputs | Distance-like comparison; a negative value means the pair is cheaper to build together |
| Pairwise joint assembly space | {func}`~assemblytheorytools.assembly.calculate_assembly_index_pairwise_joint` | List of NetworkX graphs | {class}`~networkx.DiGraph` composing every pair's pathway | Building the shared assembly space of a whole set |
| Assembly ratio | {func}`~assemblytheorytools.assembly.calculate_assembly_index_ratio` | Graph plus a settings dictionary | `float`, edges ÷ index | Size-normalised comparison: how much reuse each step buys |
| Joining-operation ratio | {func}`~assemblytheorytools.assembly.calculate_assembly_index_jo_ratio` | Graph plus a settings dictionary | `float`, edges ÷ joining operations | The same normalisation against the joining-operation index |
| Many indices at once | {func}`~assemblytheorytools.assembly.calculate_assembly_index_parallel` | List of graphs plus a settings dictionary | Lists of indices, virtual objects and pathways | Datasets and sweeps; see {doc}`guide/parallel` |

## Bounds and cheap estimates

Every entry here is analytic or tabulated: none of them runs the search, so all
are effectively instant. Use them to screen a dataset before committing to
exact calculations.

| Quantity | Function | Input | Output | Use it for |
| --- | --- | --- | --- | --- |
| Upper bound | {func}`~assemblytheorytools.assembly.calculate_assembly_index_upper_bound` | NetworkX `Graph` or RDKit `Mol` | `int`, the edge count minus one | The worst case: building with no reuse at all |
| Lower bound | {func}`~assemblytheorytools.assembly.calculate_assembly_index_lower_bound` | NetworkX `Graph` or RDKit `Mol` | `int`, from a shortest addition chain, or $\log_2$ for large graphs | The best case: maximal reuse at every step |
| Shortest addition-chain length $l(n)$ | {func}`~assemblytheorytools.assembly.calculate_integer_chain` | Integer $n$, from 1 to 9999 | `int` | The tabulated lookup behind the lower bound |
| Approximate string index | {func}`~assemblytheorytools.assembly.calculate_string_assembly_index` with `mode="cfg"` | `str` or `list[str]` | `(upper bound, virtual objects, pathway)` | Long strings, where the exact calculation is out of reach |

## Quantities measured from spectra

These estimate an assembly index from experimental data, without knowing the
structure — which is what makes the index usable as a biosignature.

| Quantity | Function | Input | Output | Use it for |
| --- | --- | --- | --- | --- |
| Recursive MA | {class}`~assemblytheorytools.recursive_ma.MAEstimator` and {func}`~assemblytheorytools.recursive_ma.rma_estimate_ma` | Fragmentation tree and a molecular weight | {class}`~numpy.ndarray` of Monte Carlo samples; report mean and spread | Estimating MA from tandem MS for an unidentified sample |
| Fragmentation tree | {func}`~assemblytheorytools.recursive_ma.rma_build_tree` | Processed MSⁿ levels as DataFrames | Nested dictionary keyed by *m/z* | Turning instrument output into the estimator's input |
| Peak count in a spectral window | {func}`~assemblytheorytools.tools_data.find_n_peak_indices_in_range` | Spectrum and a wavenumber range | `int` | Counting fingerprint-region peaks, the feature the IR model uses |
| Assembly index from IR peaks | {func}`~assemblytheorytools.tools_data.estimate_ai_from_ir_peaks` | Peak counts, reference indices and a model function | Fitted parameters and predicted indices | Calibrating and applying the IR correlation |

## Alternative complexity scores

Assembly index is one measure among many, and
{mod}`assemblytheorytools.complexity_scores` implements the usual alternatives
so they can be compared on the same molecules. Most take an RDKit `Mol` and
return a `float`. See {doc}`guide/complexity` for the full list.

| Quantity | Function | Input | Output | Use it for |
| --- | --- | --- | --- | --- |
| Graph-theoretic indices | {func}`~assemblytheorytools.complexity_scores.bertz_complexity`, {func}`~assemblytheorytools.complexity_scores.bottcher`, {func}`~assemblytheorytools.complexity_scores.wiener_index`, {func}`~assemblytheorytools.complexity_scores.balaban_index`, {func}`~assemblytheorytools.complexity_scores.randic_index`, {func}`~assemblytheorytools.complexity_scores.kirchhoff_index` | RDKit `Mol` | `float` | Benchmarking assembly index against established topological measures |
| Shape and substructure | {func}`~assemblytheorytools.complexity_scores.spacial_score`, {func}`~assemblytheorytools.complexity_scores.proudfoot`, {func}`~assemblytheorytools.complexity_scores.mc1`, {func}`~assemblytheorytools.complexity_scores.mc2` | RDKit `Mol` | `float` | Complexity that topology alone misses |
| Compression proxies | {func}`~assemblytheorytools.complexity_scores.compression_zlib_smi`, {func}`~assemblytheorytools.complexity_scores.compression_ratio_zlib_graph`, {func}`~assemblytheorytools.complexity_scores.shannon_entropy` | RDKit `Mol`, NetworkX graph, or a string | `float` | A crude redundancy measure to contrast with reuse-based ones |
| Structural similarity | {func}`~assemblytheorytools.complexity_scores.tanimoto_similarity`, {func}`~assemblytheorytools.complexity_scores.dice_morgan_similarity` | Two RDKit `Mol` objects | `float` | Comparing fingerprint similarity against shared-assembly scores |
| Size counts | {func}`~assemblytheorytools.complexity_scores.count_bonds`, {func}`~assemblytheorytools.complexity_scores.count_non_h_bonds`, {func}`~assemblytheorytools.complexity_scores.molecular_weight` | RDKit `Mol` | `int` or `float` | Controlling for size when complexity is the variable of interest |

## Exploring assembly space

| Quantity | Function | Input | Output | Use it for |
| --- | --- | --- | --- | --- |
| One join up | {func}`~assemblytheorytools.neighborhood_enumeration.enumerate_up` | Two labelled graphs | List of joined graphs, before deduplication | Generating {term}`assembly possible` from a pair of structures |
| One join down | {func}`~assemblytheorytools.neighborhood_enumeration.enumerate_down` | One labelled graph | Every partition into two connected subgraphs | Deconstructing an object into candidate parts |
| Full neighbourhood | {func}`~assemblytheorytools.neighborhood_enumeration.enumerate_neighborhood` | List of labelled graphs | Dictionary of the neighbouring graphs, unique up to isomorphism, with the up and down joins that reach them | Stepwise expansion of an assembly space; see {doc}`guide/enumeration` |
| Reaction-template products | {func}`~assemblytheorytools.reassembler.assemble` and {func}`~assemblytheorytools.reassembler.origami` | One or two RDKit `Mol` objects | Product molecules, or `None` when no template applies | Chemically plausible expansion; see {doc}`guide/reassembly` |
| Generated molecule space | {class}`~assemblytheorytools.reassembler.MoleculeSpace` | Starting molecules and generation limits | A searchable pool of generated molecules | Multi-step reassembly runs |
| Alternative virtual objects | {func}`~assemblytheorytools.find_other_paths.all_shortest_paths` | RDKit `Mol` | List of virtual-object SMILES | Sampling the intermediates other shortest pathways use; it does not return pathways |

## See also

* {doc}`concepts` — the vocabulary these quantities are built from.
* {doc}`glossary` — formal definitions and symbols.
* {doc}`configuration` — the settings dictionary, environment variables and
  backend options that the calculations share.
* {doc}`modules` — the complete API reference.
