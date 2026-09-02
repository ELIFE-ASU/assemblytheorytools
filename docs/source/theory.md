# Assembly theory

This page is background: what assembly theory claims, and which of its
quantities ATT can actually compute. It summarises Sharma *et al.* (2023),
[*Assembly theory explains and quantifies selection and evolution*](https://doi.org/10.1038/s41586-023-06600-9),
which is the reference for everything below and is cited in full under
{doc}`citing`. {doc}`glossary` gives the formal definitions; {doc}`concepts` is
the shorter, API-facing version of the same material.

The motivating problem is that physics has no way to tell a complex object that
was *selected* from one that arose by chance. Evolutionary theory explains why
some things exist and others do not, but it presupposes the objects it selects
among. Assembly theory attacks this by changing what an object is taken to be:
not a point particle with a state, but an entity defined by the history that
produced it.

## Objects and assembly units

An {term}`object` in assembly theory is finite, distinguishable, persists over
time, and is *breakable*, in the sense that the set of constraints needed to
build it from elementary parts can be counted. That last clause is the load
bearing one. Standard physics treats its objects as fundamental and unbreakable;
assembly theory treats an object as anything that can be taken apart and put
back together, which is what makes its construction history measurable.

The {term}`assembly units` are the elementary building blocks the object is
broken down to. The choice is not arbitrary — it has to correspond to an
operation that can physically be caused to happen. For molecules that operation
is bond formation, so the assembly units are bonds, and an assembly index counts
bond-forming steps over the molecular graph. For strings it is concatenation of
characters; for a general labelled graph, the addition of an edge.

## The assembly index

The {term}`assembly index` $a$ is the number of steps along a shortest
{term}`path` from the assembly units to the object, where every intermediate
that has already been constructed may be reused at no further cost. It assumes
construction is serial: one joining operation at a time.

Reuse is the whole point. A long object with no internal repetition costs about
as many steps as it has parts, while one built from a repeated motif costs far
fewer, because the motif is paid for once and then reused. The index therefore
measures how much of the object is *not* explained by its own repeated
structure.

Two properties matter for what follows. The index is a property of the object,
not of any particular path — several shortest paths usually exist, and they all
give the same index. And it is computable in finite time for any finite object,
which distinguishes it from Kolmogorov complexity: the assembly index is defined
over physically realisable operations rather than over programs for a universal
computer, so it is both computable and physically interpretable. For molecules
it is also *measurable*, having been inferred experimentally from MS/MS, NMR and
infrared spectroscopy without computing it at all.

{func}`~assemblytheorytools.assembly.calculate_assembly_index` returns it, along
with the {term}`virtual objects` — the intermediates on the path — and the path
itself. See {doc}`guide/molecules`.

## Assembly depth

{term}`Assembly depth` $d$ counts the same construction assuming the joining
operations run in *parallel* rather than in series. Steps that do not depend on
each other happen at once, so the depth is the length of the longest chain of
dependencies rather than the total number of operations.

Index and depth are properties of different things: the index belongs to the
object, while a depth belongs to a specific path. Adenine is the standard
illustration — seven sequential steps, but only five when independent steps are
allowed to proceed concurrently:

```python
import networkx as nx
import assemblytheorytools as att

graph = att.smi_to_nx("Nc1ncnc2[nH]cnc12")            # adenine
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

ordered = nx.DiGraph()
ordered.add_nodes_from((n, pathway.nodes[n]) for n in nx.topological_sort(pathway))
ordered.add_edges_from(pathway.edges(data=True))
att.assign_levels(ordered)

print(ai)                                              # 7
print(max(d["level"] for _, d in ordered.nodes(data=True)))   # 5
```

The topological re-ordering is required;
{func}`~assemblytheorytools.construction.assign_levels` raises `KeyError` on the
pathway as returned. {doc}`guide/pathways` explains why.

The depth printed above belongs to the returned shortest-*index* pathway, and
the path that minimises depth is generally not the path that minimises the
index. The separate
{func}`~assemblytheorytools.assembly.calculate_assembly_depth_rust` search
returns the object's minimum achievable depth, though it is substantially more
expensive and has no timeout.

## Copy number and the assembly equation

A single complex object is weak evidence of anything. Given enough time and
material, a combinatorial process will eventually produce almost any particular
structure once. What a random process will not do is produce the *same* complex
structure many times over, because the space of possibilities grows
super-exponentially with assembly index and any one target becomes correspondingly
unlikely.

So the assembly index alone cannot detect selection. It has to be paired with
{term}`copy number` $n$, the number of copies of the object actually observed.
High index *and* high copy number together imply a mechanism that produces that
object repeatedly — which is what selection means here.

The two combine in the assembly equation, which quantifies the selection needed
to produce a whole ensemble:

$$A = \sum_{i=1}^{N} e^{a_i}\left(\frac{n_i - 1}{N_\mathrm{T}}\right)$$

$N$ is the number of unique objects, $a_i$ and $n_i$ the assembly index and copy
number of object $i$, and $N_\mathrm{T}$ the total number of objects in the
ensemble. The exponential makes assembly grow sharply with index, the $n_i - 1$
factor discards objects seen only once, and dividing by $N_\mathrm{T}$ lets
ensembles of different sizes be compared.

{func}`~assemblytheorytools.assembly.calculate_assembly` evaluates it. Copy
numbers are an input — ATT computes the indices, but the $n_i$ come from your
measurement:

```python
graphs = [att.smi_to_nx("CCO"),                                  # ethanol,  a = 1
          att.smi_to_nx("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]         # caffeine, a = 9

att.calculate_assembly(graphs, [100.0, 100.0],
                       settings={"strip_hydrogen": True},
                       parallel=False)                           # 4012.37

att.calculate_assembly(graphs, [100.0, 1.0],
                       settings={"strip_hydrogen": True},
                       parallel=False)                           # 2.66
```

The second call is the point of the equation in miniature. Caffeine is the same
molecule with the same index in both, but as a single copy its $n_i - 1$ factor
is zero and it contributes nothing at all. Only ethanol's hundred copies count,
and assembly collapses by three orders of magnitude.
{func}`~assemblytheorytools.assembly.calculate_string_assembly` is the string
equivalent.

`parallel=False` keeps the example runnable anywhere; two molecules are not
worth a process pool. The default is `parallel=True`, which needs an
`if __name__ == "__main__":` guard when called from a script — see
{doc}`guide/parallel`.

## Assembly spaces

The {term}`assembly space` of an object is the set of virtual objects and
joining operations describing how it is built. The {term}`assembly pool` is what
is available to build with at a given moment: the assembly units plus every
virtual object made so far. A {term}`joint assembly space` does this for several
objects at once, sharing intermediates between them — which is why joint indices
fall below the sum of the separate ones, and why that gap measures how much
structure the objects have in common. ATT computes it exactly by passing a
disconnected graph to the calculator, or approximately by merging separate
pathways with {class}`~assemblytheorytools.reassembler.MoleculeSpace`.

Four spaces nest inside one another, each a constrained version of the last:

{term}`Assembly universe` ($A_U$)
: Everything the units can combine into with no rules at all. Mathematically
  well defined and physically useless: it grows doubly exponentially, has no
  ordering in time, and for most systems contains vastly more objects than there
  is matter in the observable universe.

{term}`Assembly possible` ($A_P$)
: The subspace left once physically impossible joins are removed, with every
  rule available at every step to every object. This is forward combinatorial
  expansion under physical constraints — what
  {func}`~assemblytheorytools.neighborhood_enumeration.enumerate_up` and
  {mod}`assemblytheorytools.reassembler` explore. See {doc}`guide/enumeration`.

{term}`Assembly contingent` ($A_C$)
: The subspace where history matters: only constraints already used on a path
  are available later on that path. This is where assembly theory departs from
  ordinary combinatorics, and it is a much smaller space — the past restricts
  the future.

{term}`Assembly observed` ($A_O$)
: What is actually measured, typically in high copy number. It is reconstructed
  by breaking observed objects down to their units and rebuilding minimal paths,
  and is represented by the joint assembly space.

Everything ATT computes lives in the last of these. Give the calculator a
molecule and it recovers a minimal construction for something you already have
in hand.

## Timescales, selectivity and selection

Whether selection can appear at all depends on the relation between two rates.
The {term}`discovery timescale` $\tau_d$ is how long it takes for genuinely new
objects to be found; the {term}`production timescale` $\tau_p$ is how long it
takes to make more copies of ones already discovered. (The
{term}`persistence timescale` $\tau_l$ is how long an object lasts before
transforming, and bounds how long historical contingency can be sustained at
all.)

Discovery is modelled as

$$\frac{\mathrm{d}N_{a+1}}{\mathrm{d}t} = k_\mathrm{d}(N_a)^\alpha$$

where $N_a$ is the number of objects at assembly index $a$ and $k_\mathrm{d}$ is
the rate of discovery. The exponent $\alpha$ carries the argument. At
$\alpha = 1$ every object built in the past is available for reuse and
exploration is {term}`undirected <Undirected exploration>` — history without
selection, homogeneous expansion, copy numbers spread thinly across a
combinatorial explosion of one-off objects. For $0 \le \alpha < 1$ only a subset
is reused, exploration becomes {term}`directed <Directed exploration>`, and the
system pushes deeper into the space along a few preferred paths at the cost of
exploring less of it.

That transition from undirected to directed exploration is
{term}`selectivity`. Its signature is a *lower* exploration ratio at *higher*
complexity: fewer of the reachable objects actually realised, and the ones that
are, more assembled.

{term}`Selection` is what selectivity produces when the timescales cooperate.
If $\tau_d \ll \tau_p$, new objects appear faster than resources can copy any of
them, and the result is a combinatorial explosion of low-copy-number junk — the
formose reaction and prebiotic tars are the standard examples. If
$\tau_p \ll \tau_d$, resources go into copying what already exists and little new
is found. Selection needs the two comparable, $\tau_d \approx \tau_p$: new
objects discovered often enough to matter, and copied often enough to persist.

For ATT the practical consequence is a boundary on what the package can tell
you. Selectivity depends only on discovery and carries no notion of copy number,
so an assembly index — or a whole distribution of them — speaks to selectivity.
Selection depends on copy numbers too, and ATT does not measure those; they have
to come from your experiment and be supplied to
{func}`~assemblytheorytools.assembly.calculate_assembly`. Without them, the
honest claim is about selectivity, not selection.

## Reading on

* {doc}`glossary` — the formal definitions of every term used here.
* {doc}`concepts` — the same vocabulary, restricted to what the API exposes.
* {doc}`citing` — the papers, including the algorithm behind the calculator.
