Examples
========

Every page links to source shipped in the repository's ``examples/`` directory.
Some advanced workflows need an external dataset, optional dependency or HPC
environment; each page calls that out where applicable.

The **protocols** follow the workflows behind published results end to end and
are the best starting point for a real workflow. Each is a Jupyter notebook, rendered here
from its committed outputs; "Edit on GitHub" opens the notebook in the
repository. The **basic** examples are minimal single-purpose scripts, and the
**advanced** ones cover larger or more specialised use cases.

:doc:`protocol_1` — *Calculating assembly indices.* Molecular and string
assembly indices, individually and for a joint system, with their pathways
plotted. Needs no external data.

:doc:`protocol_2` — *Large-scale molecular assembly analysis.* Assembly index
against molecular weight across CBRDB and PubChem samples, computed in
parallel. Needs network access and is the longest-running protocol.

:doc:`protocol_3` — *Correlating assembly with IR spectroscopy.* Fits a linear
model predicting assembly index from infrared peak counts. Needs the external
Chemotion IR archive.

:doc:`protocol_4` — *Estimating assembly from tandem mass spectrometry.*
Recovers a compound's assembly index from its MS/MS spectra alone, without its
structure. The sample data is bundled.

:doc:`protocol_5` — *Copy number, abundance and ensemble assembly.* Ensemble
assembly and the exploration ratio under a simulated selection pressure, and
how to turn measured abundance into copy numbers. Simulated throughout.

.. toctree::
   :maxdepth: 1
   :caption: Protocols
   :hidden:

   protocol_1
   protocol_2
   protocol_3
   protocol_4
   protocol_5

.. toctree::
   :maxdepth: 1
   :caption: Example scripts

   Basic examples <basic>
   advanced
