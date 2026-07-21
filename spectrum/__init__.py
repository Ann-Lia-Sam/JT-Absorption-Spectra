"""Disorder-averaged vibronic-cavity absorption spectrum package.

Modularized from the notebook ``Spectrum_symm_avgdisorder_2.2``. See ``main.py``
for the Nv sweep entry point.
"""

# Pin BLAS/OpenMP to one thread per process *before* any submodule imports NumPy.
# On this 40-core box, an unpinned dense ``eigh`` spawns a thread per core and
# runs dramatically slower (thread-barrier / memory-bound oversubscription);
# diagonalizations are instead parallelized across processes (cfg.n_workers).
# OpenBLAS reads these at load time, so this must run before ``import numpy``.
# ``setdefault`` lets an explicit user-exported value still win. Importing the
# package triggers this __init__ before any ``from .<module>`` pulls in NumPy.
import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")
