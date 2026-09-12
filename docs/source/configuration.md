# Configuration

## Environment variables

ATT reads the following variables. None is required, but setting `ASS_PATH`
avoids the on-demand build of the C++ calculator described below.

`ASS_PATH`
: Full path to the `ParallelAssemblyCpp` executable (named `AssemblyCpp` in
  older upstream revisions and in ATT's own cache), which computes molecule,
  graph and string assembly indices. If unset,
  {func}`~assemblytheorytools.assembly.add_assembly_to_path` searches `PATH`
  for `ParallelAssemblyCpp` (or the older `AssemblyCpp`),
  then ATT's cache directory, and finally builds the calculator with
  {func}`~assemblytheorytools.assembly.build_assembly_cpp`. Whatever it finds
  is stored in this variable for the current Python process. Set it to use your
  own build — for example an [optimised build](install.md#optional-a-faster-parallelassemblycpp-build):

  ```bash
  export ASS_PATH=$HOME/parallelassemblycpp/build/performance/ParallelAssemblyCpp
  ```

`ASS_STR_PATH`
: Full path to a `ParallelAssemblyCpp` executable to use for *string* calculations
  instead of the one in `ASS_PATH`. One executable handles both, so this is
  only needed to compare two builds; it falls back to `ASS_PATH` when unset.

`ATT_ASSEMBLYCPP_REF`
: Branch, tag or commit of
  [parallelassemblycpp](https://github.com/ELIFE-ASU/parallelassemblycpp) that
  {func}`~assemblytheorytools.assembly.build_assembly_cpp` builds. Defaults to
  `main`. A cached build is reused only for the same ref. To fetch new commits
  on that ref, call `build_assembly_cpp(force=True)`. Explicit executable paths
  and executables on `PATH` take precedence; unset `ASS_PATH` before changing
  this variable in an existing Python process.

`XDG_CACHE_HOME`
: Standard cache location, honoured when choosing where to build and look for
  the calculator. Defaults to `~/.cache`, giving
  `~/.cache/assemblytheorytools/assemblycpp/bin/AssemblyCpp`.

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

## The C++ calculator

The distribution ships no parallelassemblycpp binary. parallelassemblycpp is licensed
CC BY-NC 4.0, which is more restrictive than this package's MIT licence, and a
prebuilt binary would in any case only serve one platform. Instead, the first
calculation that needs it runs
{func}`~assemblytheorytools.assembly.build_assembly_cpp`, which clones
[parallelassemblycpp](https://github.com/ELIFE-ASU/parallelassemblycpp), builds it and
installs the executable into ATT's cache directory. That takes a few minutes and
needs `git`, CMake 3.25 or newer, and a C++20 compiler; CMake and Ninja are
installed as dependencies of this package.
CMake and Ninja beside the running Python interpreter take precedence over
tools on `PATH`, so an absolute interpreter path still uses its environment's
build dependencies.

The build is deliberate about two settings. It does not use the repository's
`release` CMake preset, which turns warnings into errors and would fail on a
compiler newer than the one parallelassemblycpp tests against, and it sets
`BUILD_TESTING=OFF`, which CMake otherwise turns on. Set `ASS_PATH` to skip the
build entirely.

Concurrent first calculations share one build under a file lock. Installation is
staged before replacing the cached executable, so an unsuccessful rebuild keeps
the previous executable usable. The source checkout is retained for updates; a
failed build also retains its build tree for inspection. The cache records its
repository and requested ref in `build.json`. Older caches without this record
remain usable until an explicit ref is requested or a rebuild is forced.

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
: A finite, non-negative wall-clock limit for the external calculator, enforced
  by ATT; `None` disables this limit. The calculator's CPU-time budget is a
  separate option, `cpp_options.runtime_ticks`, and is unlimited by default.
  On a timeout, ATT
  interrupts the calculator (SIGINT) and allows up to two seconds to write its
  best result before killing it; Windows has no equivalent interrupt for a child
  process, so the calculator is terminated at once and the bound is recovered
  from the log. The search is exponential in the worst
  case, so a large molecule can exceed any limit. When the search stops early —
  its budget ran out, it hit its enumeration cap, or it was interrupted — ATT
  returns the best upper bound the calculator reached, or `-1` if it reached
  none. ATT reads the saved result first and falls back to the log when no
  result was written. A search the calculator completes returns an exact result.
  Raise the limit for large structures, or use
  {func}`~assemblytheorytools.assembly.calculate_assembly_index_upper_bound`
  when the edge-count bound is sufficient.

`joint_corr` (default `True`)
: Apply the component-count correction for disconnected inputs. Isolated
  vertices add no bonds or joining operations and are excluded. See
  [Joint assembly](concepts.md#joint-assembly).

`exact` (default `False`)
: Require a proven minimum. When the search stopped early, return `-1` instead
  of the best upper bound found so far. The calculator reports an early stop
  explicitly, so this does not rest on the elapsed time alone.

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

`return_log_file` (default `False`)
: Return the log path as a fourth result field and retain the calculation
  directory. This applies to both graph and string calculations. Without
  `return_log_file`, `debug`, `save_dir` or a requested diagnostic output file,
  ATT removes temporary files on success and failure. Failed launches and nonzero exits raise `OSError`;
  calculator failures include the end of the log in the exception.

{func}`~assemblytheorytools.assembly.calculate_assembly_index_jo` reads the
pathway from its own calculation directory and honours these retention options
through `settings`. Its result remains `(jo, virtual_objects, pathway)`;
`settings={"return_log_file": True}` retains the directory and prints the log
location without adding a fourth result field.

C++ string mode accepts one line of ASCII text: the calculator indexes bytes
and reads each line as a separate input. Strings of one character or fewer, and
edgeless graphs, need no joining operations and return index zero without
launching a calculator; with `return_log_file=True` the fourth field is then
`None`. Single-character items are dropped from a list of strings before the
joint encoding.

`dir_code`
: Explicit path to the calculator executable, overriding `ASS_PATH`.

## C++ command-line controls

Pass an {class}`~assemblytheorytools.assembly.AssemblyCppOptions` instance as
`cpp_options` to either C++ calculation entry point. The same instance can be
reused or passed in a batch calculation's `settings` dictionary.

```python
import assemblytheorytools as att

options = att.AssemblyCppOptions(enum_max=1_000_000, pathway=False)
ai, virtual_objects, pathway = att.calculate_assembly_index(
    att.smi_to_mol("c1ccccc1"), strip_hydrogen=True,
    timeout=30, cpp_options=options,
)
# pathway=False returns the index with both remaining fields set to None.
```

Every C++ CLI control maps to the Python interface below. ATT emits compatible
older aliases for renamed flags; the table uses the current `--help` names.

| C++ option | Python control | Default and meaning |
| --- | --- | --- |
| `--runtime` | `cpp_options.runtime_ticks` | `None`: unlimited CPU time; otherwise integer `std::clock` ticks, from 0 through `2**64 - 1`. The maximum value also means unlimited. Divide by `CLOCKS_PER_SEC` (1,000,000 with glibc) to convert ticks to seconds. |
| `--enum-max` | `cpp_options.enum_max` | `None`: C++ default, currently 50,000,000. Integers from 1 through `2**31 - 1`; graph mode only. |
| `--pathway` | `cpp_options.pathway` | `True`; disable pathway computation/output with `False`. |
| `--accept-palindromes` | `cpp_options.accept_palindromes` | `False`; allow a string fragment to be reused in reverse. Native string mode only. |
| `--parallel` | `cpp_options.parallel` | `"off"`, `"auto"` or `"on"`; default `"off"`. `"auto"` can fall back to serial; `"on"` requires a compatible parallel executable. |
| `--threads` | `cpp_options.threads` | `"auto"` (default) or an integer from 1 through `2**31 - 1`; threads per C++ process, applicable to parallel graph search. An explicit count is graph mode only. |
| `--verbose` | `cpp_options.verbose` | `False`; print the parsed graph into the calculator log, independently of Python's `debug`. Graph mode only. |
| `--memory-report` | `cpp_options.memory_report` | `False`; write Linux peak memory to `memUsage`. |
| `--telemetry` | `cpp_options.telemetry` | `False`; write `INPUTTelemetry.json`. Requires a telemetry executable; graph mode only. |
| `--write-intermediate-mas` | `cpp_options.write_intermediate_mas` | `False`; write index improvements to `INPUTIntermediateMAs`. Graph mode only; requires serial search. |
| `--run-strings` | String calculation entry point and `mode` | ATT selects the appropriate input serializer, C++ mode and pathway parser together. |
| `--remove-hydrogens` | `strip_hydrogen` | ATT performs stripping in Python and disables C++ stripping so the pathway matches the input graph. |
| `--compensate-disjoint` | `joint_corr` | ATT applies the correction in Python and disables C++ compensation to avoid applying it twice. |
| `--help` | `att.get_assembly_cpp_help(dir_code=None)` | Return the selected executable's help text, including build-specific controls. |

Booleans must be `True` or `False`; numeric bounds are checked before execution.
Graph-only controls are rejected in native string mode instead of silently
ignored. `cpp_options` is unavailable for the CFG backend.

`parallel="on"` cannot be combined with native string mode, a finite C++ CPU
budget, or intermediate index output. `parallel="auto"` permits the calculator's
serial fallback for the latter two; string mode is always serial. The Python
wall-clock `timeout` works with every mode and does not force serial execution.
See {doc}`guide/parallel` for selecting a parallel build.

ATT's default build is serial and has no telemetry. Point `dir_code` or
`ASS_PATH` at `ParallelAssemblyCppOMP` for OpenMP, `ParallelAssemblyCppTelemetry`
for telemetry, or `ParallelAssemblyCppOMPTelemetry` for both, from an upstream
build configured with `PARALLELASSEMBLYCPP_BUILD_OPENMP=ON` and/or
`PARALLELASSEMBLYCPP_BUILD_TELEMETRY=ON`. Unsupported requested features raise
the calculator's error; disabled telemetry emits no flag.

Requesting a memory report, telemetry or intermediate indices retains the
calculation directory and prints its location. Use `return_log_file=True` to
retrieve that location programmatically: output files sit next to the returned
log, with `INPUT` equal to `graph_in` or `string_in`. Both graph and string
entry points also accept `save_dir=True` to retain their working files.

With `accept_palindromes=True`, the returned string pathway distinguishes
`operation="concatenate"` nodes with `cost=1` from `operation="reverse"` nodes
and edges with `cost=0`. Summing **node** costs counts joining operations;
counting all nonprimitive nodes would also count free reversals. Without it the
pathway carries no `operation` or `cost` attribute at all, and every
nonprimitive node is one join.

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
  most that many, `0` for every minimum pathway the search actually discovered,
  or `None` to skip reconstruction. `0` is not an exhaustive enumeration:
  bounding and memoisation prune pathways that merely tie the minimum.
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
* Every node needs a nonempty string `color` attribute without whitespace — the element
  symbol for molecules, any label for arbitrary graphs.
* Every edge needs an integer `color` attribute from 1 through 32767 (bond order
  for molecules). NumPy integers are accepted; strings, floats and booleans
  are rejected.
* Graphs must be simple and undirected, with no self loops and at most 32767
  vertices. The graph name must fit on one line.

Invalid graphs raise `ValueError` before the calculator is invoked.

{func}`~assemblytheorytools.tools_graph.smi_to_nx`,
{func}`~assemblytheorytools.tools_cell.cif_to_nx` and
{func}`~assemblytheorytools.tools_cell.cell_to_nx` produce conforming graphs. See
{doc}`guide/graphs` for building one by hand.
