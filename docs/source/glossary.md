# Glossary

The terminology used here follows the assembly-theory literature cited under
{doc}`citing`. ATT-specific notes and cross-references are added where the API
implements a term directly.

{doc}`theory` puts these terms into a narrative; {doc}`concepts` covers the
smaller set of them that the API surfaces directly.

## Basic assembly variables

:::{glossary}
Object
  *Symbol: $O$.* An entity that is finite, distinguishable, persists over time,
  and is breakable such that the set of constraints to construct it from
  elementary building blocks is quantifiable/measurable. This set of constraints
  (physical or informational) refers to the contingent, recursive relationships
  linking the construction steps to each other. Note that, typically, the
  observed persistence time of objects undergoing selection is much longer than
  their isolated half-life because they are subject to copying.

Assembly units
  The elementary building blocks from which the object is constructed.

  In ATT: bonds, for a molecular graph; characters, for a string; edges, for an
  arbitrary labelled graph.

Path
  A sequence of joining operations.

  In ATT: the third value returned by
  {func}`~assemblytheorytools.assembly.calculate_assembly_index`, as a
  {class}`~networkx.DiGraph`. See {doc}`guide/pathways`.

Assembly index
  *Symbol: $a$.* The number of steps along any shortest path required to
  construct an object from its basic assembly units. It assumes that the
  construction processes are serial or sequential. This is a measurable
  parameter (empirically inferred through NMR, MS/MS and IR spectroscopy
  experiments for molecular objects) and can only be determined by recursive
  deconstruction of the object in isolation. For molecules, the assembly index
  (or, molecular assembly, MA) is calculated/measured with respect to the
  fundamental units observed, i.e., bonds.

  In ATT: the first value returned by
  {func}`~assemblytheorytools.assembly.calculate_assembly_index`.

Copy number
  *Symbol: $n$.* Corresponds to the number of copies of an observed (measured)
  object in a physical system.

  In ATT: copy numbers are an input, not something the package measures — they
  are the `n_i` argument of
  {func}`~assemblytheorytools.assembly.calculate_assembly`.
  {func}`~assemblytheorytools.assembly.count_copies` turns a list of
  observations into the unique objects and counts that argument expects.

Assembly
  *Symbol: $A$.* An integrated quantity over an ensemble of observed objects. It
  is summed over all unique objects, where each object assembly is equal to
  $e^{a}(n-1)$ divided by the total number of objects $N_\mathrm{T}$. It aims to
  quantify the total amount of selection associated with the required
  constraints to produce an ensemble of observed objects.

  In ATT: {func}`~assemblytheorytools.assembly.calculate_assembly`, and
  {func}`~assemblytheorytools.assembly.calculate_string_assembly` for strings.
  {func}`~assemblytheorytools.assembly.calculate_assembly_from_indices`
  evaluates the same equation from indices that are already known.

Assembly depth
  *Symbol: $d$.* The number of steps along a path required to construct the
  object from its basic assembly units. It assumes that the construction
  processes are parallel or concurrent. Note that a given assembly depth is a
  property of the associated path while the assembly index (only associated with
  the shortest path) is a property of the object. The shortest assembly depth is
  usually not associated with the assembly index path.

  In ATT: {func}`~assemblytheorytools.construction.assign_levels` annotates each
  pathway node with the depth of the pathway it belongs to, while
  {func}`~assemblytheorytools.assembly.calculate_assembly_depth_rust` returns the
  object's minimum achievable assembly depth. See {doc}`guide/pathways`.

Virtual objects
  *Symbol: $o_v$.* The contingent sub-objects along a given assembly path. In
  other words, they are the intermediate sub-objects of the construction process
  between the assembly units and the global object of interest. Note that
  virtual objects are not necessarily measurable/observable.

  In ATT: the second value returned by
  {func}`~assemblytheorytools.assembly.calculate_assembly_index`: graphs for a
  NetworkX input, or SMILES strings for an RDKit `Mol` input.

Virtual copy number
  *Symbol: $n_v$.* Is defined for the virtual objects along the assembly path,
  which is particularly useful within the context of a joint assembly space
  where multiple objects coexist. It also quantifies the efficiency of the
  construction process in the joint assembly space based on the contribution of
  the sub-objects in constructing the observed object.

Assembly space
  Corresponds to the set of (virtual) objects and joining operations that
  describe the construction process of one given object.

Joint assembly space
  Corresponds to the set of (virtual) objects and joining operations that
  describe the construction process of an ensemble of objects. Due to its
  computationally intensive nature, it is sometimes approximated by the union of
  individual assembly spaces of all the observed objects.

  In ATT: pass a disconnected graph to
  {func}`~assemblytheorytools.assembly.calculate_assembly_index` for the exact
  quantity, or merge separate pathways with
  {func}`~assemblytheorytools.assembly.joint_assembly_space` or
  {class}`~assemblytheorytools.reassembler.MoleculeSpace` for the union
  approximation.

Assembly pool
  Corresponds to all the entities available for use in the construction process
  of an object, i.e., both the basic assembly units and the generated virtual
  objects along previous steps along the path.

  In ATT:
  {class}`~assemblytheorytools.reassembler.MoleculeGenerationAssemblyPool`
  represents one generation of a pool. See {doc}`guide/reassembly`.
:::

## Assembly spaces

:::{glossary}
Assembly universe
  *Symbol: $A_U$.* Represents the space constructed from elementary units
  without any constraints on the combinational rules.

Assembly possible
  *Symbol: $A_P$.* Represents the space of physically plausible objects by
  combinatorial expansion constrained by the physical rules of object
  construction and allowing all rules to be available at every step for every
  object. In other words, it is a sub-space of the assembly universe where all
  the objects constructed from unphysical joining operations have been removed.

  In ATT: {func}`~assemblytheorytools.neighborhood_enumeration.enumerate_up`
  and {mod}`assemblytheorytools.reassembler` both expand forwards under
  physical constraints. See {doc}`guide/enumeration`.

Assembly contingent
  *Symbol: $A_C$.* Represents the space of physically plausible objects where
  selection on the history matters. It is a sub-space of the assembly possible
  where historical contingency is introduced by the assumption that only the
  constraints used on a specific path can be used in the future.

Assembly observed
  *Symbol: $A_O$.* Represents the space of the observed objects, which is a
  subset of assembly contingent. The observed objects are usually experimentally
  measured and present in higher copy numbers. Assembly observed is
  reconstructed by breaking the observed objects apart to their elementary
  building blocks and reconstructing a minimum path to construct those objects.
  $A_O$ is represented by the joint assembly space.
:::

## Assembly characteristic timescales

:::{glossary}
Persistence timescale
  *Symbol: $\tau_l$.* Is the characteristic timescale for an object to last
  before transforming into other objects. Within an assembly space, the
  persistence timescale also represents the characteristic timescale up to which
  the historical contingency can be sustained.

Discovery timescale
  *Symbol: $\tau_d$.* Is the characteristic discovery timescale at any assembly
  index which quantifies the timescale at which new unique objects get
  discovered.

Production timescale
  *Symbol: $\tau_p$.* Is the characteristic timescale defined by the rate of
  production of objects by a physical process. This is governed by mass transfer
  kinetics of the construction process ($\tau_p \sim 1/\kappa_p$), where
  $\kappa_p$ is the production rate.
:::

## Selection in assembly space

:::{glossary}
Contingency
  Is a property of a non-Markovian process and corresponds to the effect of
  finite knowledge being transmitted from one discrete time step to another.

Undirected exploration
  Describes the process by which exploration is random in assembly space and
  corresponds to assembly possible. It is consistent with homogeneously
  distributed copy numbers.

Directed exploration
  Describes the process by which exploration is non-random in assembly space and
  corresponds to assembly contingent. The recursive operations to build objects
  exhibit goal-directedness and more complexity is achieved in the system at the
  cost of exploring "less" diversity-wise.

Selectivity
  Describes the exploratory power of the assembly space and characterizes the
  transition of a system from undirected exploration to directed exploration. In
  other words, it is the outcome of the emergence of preferential paths among
  all possible paths. Selectivity only depends on the discovery timescale and
  does not include any notion of copy number. In addition, it does not include
  selectivity among the basic assembly units from which objects are built, but
  only represents (virtual) object-level interactions.

  The assembly contingent space can include selected objects that have been
  discovered along permitted histories but have not yet been produced or
  observed.

Exploration ratio
  The fraction of a {term}`joint assembly space` that was actually observed:
  the number of observed objects divided by the total, which also counts the
  contingent objects that were never observed but are required to construct
  the observed ones along a minimum path. A ratio near one indicates
  {term}`undirected exploration`, since the system realised nearly everything
  its own construction implies; a markedly lower ratio indicates
  {term}`directed exploration`, since most of the reachable space was left
  unrealised. Unlike {term}`assembly`, it carries no notion of copy number, so
  it reports {term}`selectivity` rather than {term}`selection`.

  In ATT: {func}`~assemblytheorytools.assembly.exploration_ratio`, measured
  over the space built by
  {func}`~assemblytheorytools.assembly.joint_assembly_space`.

Complexity-diversity space
  Any physical temporal process can be represented in complexity-diversity space
  for quantifying the degree of selectivity, where complexity is quantified by
  the assembly index.

Selection
  Is the result of selectivity in a system, and represents the subset of objects
  in the assembly contingent space which are observed with high copy numbers. It
  depends on both discovery and production timescales. Therefore, in the absence
  of explicit quantification of the copy number of the observed objects, we can
  only assess selectivity, not selection. In a physical system, this
  goal-directed exploration maps to dynamics of cooperation and competition.

  Note that when a system maintains / remains in directed exploration, we talk
  about persistence.
:::
