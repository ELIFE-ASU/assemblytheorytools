# Configuration

## Environment variables

ATT reads the following variables. None is required for a default install — the
package ships precompiled calculators and falls back to them.

`ASS_PATH`
: Directory holding the assemblyCPP executable used for molecule and graph
  calculations. If unset,
  {func}`~assemblytheorytools.assembly.add_assembly_to_path` locates the
  binary bundled in `assemblytheorytools/precompiled/`, compiling one if
  necessary, and sets the variable for the rest of the session. Set it to use
  your own build — for example an
  [oneAPI build](install.md#optional-a-faster-assemblycpp-with-intel-oneapi):

  ```bash
  export ASS_PATH=$HOME/assemblycpp-v5/v5/asscpp
  ```

`ASS_STR_PATH`
: The same, for the *string* calculator. Resolved by the same helper when
  `str_mode=True`, and backed by a separate bundled binary. Only set it if you
  have built the string calculator yourself.

`ORCA_PATH`
: Full path to the ORCA executable, including the binary name. Read by the
  quantum-chemistry helpers in {mod}`assemblytheorytools.tools_atoms`
  ({func}`~assemblytheorytools.tools_atoms.orca_calc_preset`,
  {func}`~assemblytheorytools.tools_atoms.optimise_atoms`,
  {func}`~assemblytheorytools.tools_atoms.calculate_ccsd_energy`,
  {func}`~assemblytheorytools.tools_atoms.calculate_free_energy` and
  friends). Defaults to `orca`, i.e. whatever is on `PATH`. Not used by any
  assembly index calculation.

  ```bash
  export ORCA_PATH=$HOME/orca_6_1_1/orca
  ```

`CP2K_COMMAND`
: Command used to launch CP2K from
  {func}`~assemblytheorytools.tools_atoms.cp2k_calc_preset`. Defaults to
  `cp2k.popt`.

Every function that reads one of these also takes an explicit path argument,
which wins over the environment. Prefer the argument in library code and the
variable in interactive or batch use.

{func}`~assemblytheorytools.assembly.add_assembly_to_path` can persist the
setting for you by appending an `export` line to `~/.bashrc` and `~/.profile`;
it only does so when asked, since it edits files outside the project.

## Bundled binaries

The wheel ships three static Linux builds under
`assemblytheorytools/precompiled/`:

| File | Used for |
| --- | --- |
| `asscpp_combined_static_linux` | Molecule and graph assembly indices |
| `asscpp_combined_static_strings` | String assembly indices |
| `asscpp_public_static_linux` | Public build, used as a fallback |

They are Linux x86-64 binaries. On other platforms, build assemblyCPP from
source and set `ASS_PATH`.

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
  the worst case, so a large molecule can exceed any limit; on timeout the
  calculation is abandoned rather than returning a partial answer. Raise it for
  big structures, and prefer
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_upper_bound`
  when an approximate answer will do.

`joint_corr` (default `True`)
: Apply the component-count correction for disconnected inputs. See
  [Joint assembly](concepts.md#joint-assembly).

`exact` (default `False`)
: Force the calculator's exact mode.

`canonicalize` (default `True`)
: Canonicalise node labels before writing the calculator input, so that two
  isomorphic graphs with different node numbering give the same result.

`debug` (default `False`) and `save_dir` (default `False`)
: Keep the temporary working directory and print the calculator's output.
  Useful when a calculation fails or returns a surprising index; the directory
  holds the generated input file and the calculator's raw log.

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
  backend itself takes milliseconds. Unlike the C++ backend, a timed-out search
  still returns an answer — an upper bound — with `states_searched` set to
  `None` to say so.

`canonize` (default `'tree-nauty'`)
: Canonisation mode: `'nauty'`, `'faulon'`, `'tree-nauty'` or `'tree-faulon'`.

`parallel` (default `'depth-one'`)
: Parallelisation mode: `'none'`, `'depth-one'` or `'always'`. Use `'none'` to
  make `states_searched` reproducible.

`memoize` (default `'canon-index'`)
: Memoisation mode: `'none'` or `'canon-index'`.

`kernel` (default `'none'`)
: Kernelisation mode: `'none'`, `'once'`, `'depth-one'` or `'always'`.

`bounds` (default `('int', 'matchable-edges')`)
: Branch-and-bound strategies, drawn from `'log'`, `'int'`, `'vec-simple'`,
  `'vec-small-frags'` and `'matchable-edges'`. Pass an empty sequence for an
  exhaustive search.

`max_pathways` (default `None`)
: How many minimum assembly pathways to reconstruct: a positive integer for at
  most that many, `0` for all of them, or `None` to skip reconstruction. Only
  available on releases newer than 0.6.1 — see
  [Pathways](guide/pathways.md#pathways-from-the-rust-backend).

`vo_type` (default `'smiles'`)
: Representation for the virtual objects in any reconstructed pathway:
  `'graph'`, `'mol'`, `'smiles'` or `'inchi'`.

Unlike the C++ backend, this one always strips hydrogens and does not accept
`strip_hydrogen`, `joint_corr` or `canonicalize`.

## Graph input requirements

The calculator input format constrains what the graph may contain:

* Node indices must start at 0 and be contiguous.
* Every node needs a `color` attribute — the element symbol for molecules, any
  label for arbitrary graphs.
* Every edge needs a `color` attribute that is an **integer** (bond order for
  molecules, starting at 1). A string here raises an `AssertionError` from
  {func}`~assemblytheorytools.tools_graph.write_ass_graph_file`.

{func}`~assemblytheorytools.tools_graph.smi_to_nx` and
{func}`~assemblytheorytools.tools_cell.cif_to_nx` produce conforming graphs. See
{doc}`guide/graphs` for building one by hand.
