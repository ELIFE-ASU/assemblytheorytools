# Configuration

## Environment variables

ATT reads the following variables. On Linux x86-64, none is required for the
default calculators because the package includes compatible binaries. Other
platforms must build assemblyCPP and configure its path.

`ASS_PATH`
: Full path to the assemblyCPP executable used for molecule and graph
  calculations. If unset,
  {func}`~assemblytheorytools.assembly.add_assembly_to_path` locates the
  bundled binary in `assemblytheorytools/precompiled/` and sets the variable
  for the current Python process. If the packaged binary is missing, the helper
  attempts a source build. Set it to use your own build — for example an
  [oneAPI build](install.md#optional-a-faster-assemblycpp-with-intel-oneapi):

  ```bash
  export ASS_PATH=$HOME/assemblycpp-v5/v5/asscpp
  ```

`ASS_STR_PATH`
: Full path to the *string* calculator. The same helper resolves it when
  `str_mode=True`, using a separate bundled binary. Only set it if you have
  built the string calculator yourself.

`ORCA_PATH`
: Full path to the ORCA executable, including the binary name. Read by the
  quantum-chemistry helpers in {mod}`assemblytheorytools.tools_atoms`
  ({func}`~assemblytheorytools.tools_atoms.orca_calc_preset`,
  {func}`~assemblytheorytools.tools_atoms.optimise_atoms`,
  {func}`~assemblytheorytools.tools_atoms.calculate_ccsd_energy`,
  {func}`~assemblytheorytools.tools_atoms.calculate_free_energy` and
  friends). Set it explicitly (or pass `orca_path` where supported); the
  helpers do not share a reliable `PATH`-only fallback. It is not used by any
  assembly index calculation.

  ```bash
  export ORCA_PATH=$HOME/orca_6_1_1/orca
  ```

`CP2K_COMMAND`
: Command used to launch CP2K from
  {func}`~assemblytheorytools.tools_atoms.cp2k_calc_preset`. Defaults to
  `cp2k.popt`.

The assembly entry points also take `dir_code`, which wins over `ASS_PATH` or
`ASS_STR_PATH`. Prefer that argument in library code and an environment variable
in interactive or batch use. `add_assembly_to_path` changes only the current
process; add an `export` line to your shell configuration yourself when the
setting should persist.

## Bundled binaries

The wheel ships three static Linux builds under
`assemblytheorytools/precompiled/`:

| File | Used for |
| --- | --- |
| `asscpp_combined_static_linux` | Molecule and graph assembly indices |
| `asscpp_public_static_linux` | Directed string assembly indices |
| `asscpp_combined_static_strings` | Legacy combined/string build; not selected automatically |

They are Linux x86-64 binaries. On other platforms, build assemblyCPP from
source and set `ASS_PATH`. Directed-string calculations also require
`ASS_STR_PATH`; both variables may point to one compatible combined build.

`assemblytheorytools/data/integer_chain_9999.txt` is a lookup table of
precomputed integer-chain assembly indices used by
{func}`~assemblytheorytools.assembly.calculate_integer_chain`.

## Calculation options

These arguments appear on
{func}`~assemblytheorytools.assembly.calculate_assembly_index` and most of the
functions built on it.

`strip_hydrogen` (default `False`)
: Remove hydrogens before calculating. Almost always what you want for
  molecular assembly indices — see [Hydrogens](concepts.md#hydrogens). The
  stripping is applied to a copy, so the graph you pass is left unchanged.

`timeout` (default `100.0` seconds)
: Wall-clock limit for the external calculator. The search is exponential in
  the worst case, so a large molecule can exceed any limit. With the default
  timeout handling, a timed-out calculation returns the best upper bound logged
  so far, or `-1` if it found none. A calculation that finishes still returns an
  exact result. Raise the limit for large structures, or use
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_upper_bound`
  when the edge-count bound is sufficient.

`joint_corr` (default `True`)
: Apply the component-count correction for disconnected inputs. See
  [Joint assembly](concepts.md#joint-assembly).

`exact` (default `False`)
: Require an exact result. A timed-out calculation returns `-1` instead of the
  best upper bound found so far.

`canonicalize` (default `True`)
: Relabel nodes, in their current iteration order, to contiguous integers
  starting at zero before writing the calculator input. This satisfies the
  input format; despite the historical name, it is not canonical graph
  labelling.

`debug` (default `False`) and `save_dir` (default `False`)
: Keep the temporary working directory; `debug=True` also prints ATT's Python
  diagnostics. Useful when a calculation fails or returns a surprising index;
  the directory holds the generated input file and the calculator's standard
  output/error in `assembly_output.log`.

`dir_code`
: Explicit path to the calculator executable, overriding `ASS_PATH`.

## Rust backend options

The Rust backend needs no environment variable and no binary of its own: it is
installed as the `assembly-theory` wheel and called in-process. Its search is
configured entirely through the arguments of
{func}`~assemblytheorytools.assembly.calculate_assembly_index_rust_search`.

`timeout` (default `None`)
: Seconds after which to stop searching and return the best index found so far.
  Given in seconds to match
  {func}`~assemblytheorytools.assembly.calculate_assembly_index`, although the
  backend itself takes milliseconds. A timed-out search returns its best upper
  bound, like the default C++ mode; Rust specifically sets `states_searched` to
  `None` to mark the incomplete search.

`canonize` (default `'tree-nauty'`)
: Canonisation mode: `'nauty'`, `'faulon'`, `'tree-nauty'` or `'tree-faulon'`.

`parallel` (default `'depth-one'`)
: Parallelisation mode: `'none'`, `'depth-one'` or `'always'`. Use `'none'` to
  make `states_searched` reproducible.

`memoize` (default `'canon-index'`)
: Memoisation mode: `'none'` or `'canon-index'`. The backend's error message
  also lists `'frags-index'`, but rejects that value.

`kernel` (default `'none'`)
: Kernelisation mode: `'none'`, `'once'`, `'depth-one'` or `'always'`.

`bounds` (default `('int', 'matchable-edges')`)
: Branch-and-bound strategies, drawn from `'log'`, `'int'`, `'vec-simple'`,
  `'vec-small-frags'` and `'matchable-edges'`. Pass an empty sequence for an
  exhaustive search.

`max_pathways` (default `None`)
: How many minimum assembly pathways to reconstruct: a positive integer for at
  most that many, `0` for all of them, or `None` to skip reconstruction.
  Requires `assembly-theory` 0.7.0 or newer — see
  [Pathways](guide/pathways.md#pathways-from-the-rust-backend).

`vo_type` (default `'smiles'`)
: Representation for the virtual objects in any reconstructed pathway:
  `'graph'`, `'mol'`, `'smiles'` or `'inchi'`.

Unlike the C++ backend, this one always strips hydrogens and does not accept
`strip_hydrogen`, `joint_corr` or `canonicalize`.

## Graph input requirements

The calculator input format constrains what the graph may contain. The default
`canonicalize=True` handles the first rule before writing the input; it matters
when calling {func}`~assemblytheorytools.tools_graph.write_ass_graph_file`
directly or disabling canonicalisation.

* Node indices must start at 0 and be contiguous.
* Every node needs a string `color` attribute without spaces — the element
  symbol for molecules, any label for arbitrary graphs.
* Every edge needs a `color` attribute that is an **integer** (bond order for
  molecules, starting at 1). A string here raises an `AssertionError` from
  {func}`~assemblytheorytools.tools_graph.write_ass_graph_file`.

{func}`~assemblytheorytools.tools_graph.smi_to_nx` and
{func}`~assemblytheorytools.tools_cell.cif_to_nx` produce conforming graphs. See
{doc}`guide/graphs` for building one by hand.
