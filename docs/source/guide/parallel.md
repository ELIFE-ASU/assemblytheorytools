# Parallel calculations

Assembly index calculations are independent of one another. The default C++
backend spawns an external process for each calculation, so batches parallelise
well.

## Many molecules at once

{func}`~assemblytheorytools.assembly.calculate_assembly_index_parallel` takes a
list of graphs and a settings dictionary forwarded to each call. Note that
`settings` is a required positional argument — pass `None` for the defaults:

```python
import assemblytheorytools as att

smiles = ["NCC(=O)O", "CC(N)C(=O)O", "CC", "c1ccccc1"]
graphs = [att.smi_to_nx(s) for s in smiles]

ai, virt_obj, pathway = att.calculate_assembly_index_parallel(
    graphs, dict(strip_hydrogen=True))

print(ai)   # [3, 4, 0, 3]
```

It returns three lists — indices, virtual objects and pathways — aligned with
the input order, so results stay matched to their inputs even though the
calculations finish out of order.

## The general parallel map

{func}`~assemblytheorytools.tools_mp.mp_calc` maps any function over an iterable
across processes:

```python
def ai_of(smi):
    return att.calculate_assembly_index(att.smi_to_nx(smi), strip_hydrogen=True)[0]


if __name__ == "__main__":
    print(att.mp_calc(ai_of, smiles, n=4))   # [3, 4, 0, 3]
```

:::{important}
On POSIX platforms, Python 3.14 starts worker processes with the **forkserver**
method, which pickles the target function by reference. For portable code, the
function must be importable from a real module at import time:

* define it at module level in a `.py` file — not in a closure, a `lambda`, or
  a comprehension;
* guard the entry point with `if __name__ == "__main__":`;
* run it as a script, not with `python -c`.

A function defined interactively or nested inside another function raises a
pickling error when the pool starts.
:::

Variants:

* {func}`~assemblytheorytools.tools_mp.mp_calc_chunked` — batches items before
  dispatch. Use it for large inputs where per-item dispatch overhead dominates;
  `chunksize=None` uses one item per batch.
* {func}`~assemblytheorytools.tools_mp.mp_calc_star` — for functions taking
  several arguments, mapping over tuples.
* {func}`~assemblytheorytools.tools_mp.tp_calc` — threads instead of processes.
  Only worth it for I/O-bound work such as PubChem downloads; assembly
  calculations are process-bound and should use `mp_calc`.

`n` sets the worker count and defaults to `multiprocessing.cpu_count()`. Match
it to the machine — on a shared HPC node, set it to the cores your job was
allocated, not the cores the node has.

## Timeouts at scale

Over a large dataset some molecules will exceed the calculator's time limit.
Set `timeout` in the settings dictionary so slow outliers do not stall the
batch, then filter the results:

```python
ai, virt_obj, pathway = att.calculate_assembly_index_parallel(
    graphs, dict(strip_hydrogen=True, timeout=30.0, exact=True))

ok = [(g, a) for g, a in zip(graphs, ai) if a >= 0]
```

A negative index marks a calculation that did not produce an exact answer;
zero is valid for an object with one elementary bond. Without `exact=True`, a
positive result after a timeout may be an upper bound rather than the exact
index. [Protocol 2](../examples/protocol_2.md) deliberately excludes both
failures and the trivial zero-index case from its plotted subset.

## Working on an HPC cluster

For job arrays, run one chunk of the dataset per array task and combine the
outputs afterwards rather than requesting one very large parallel job.
`examples/advanced/kegg_compounds/job_array/` contains a worked submission
script.

Two practical notes:

* Invoke Python by absolute path (`srun $HOME/.conda/envs/ass_env/bin/python3`)
  so the job uses the intended environment.
* {func}`~assemblytheorytools.tools_file.write_to_shared_file` appends to a
  shared results file with locking, which is safe when several array tasks
  write concurrently.

## See also

* {doc}`../api/tools_mp` — the parallel map helpers.
* {doc}`../api/tools_file` — file utilities for batch runs.
* {doc}`../examples/protocol_2` — a full large-scale workflow.
