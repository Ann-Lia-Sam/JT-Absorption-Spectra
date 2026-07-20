"""Shared disorder-realization generator.

The disorder-averaged absorption spectrum (:mod:`spectrum.spectrum`) draws its
per-realization electronic-energy shifts inline as::

    rng = np.random.default_rng(cfg.rng_seed)
    for r in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)

The vibronic / heatmap workflow must use the *same* disorder values so that a
given realization index refers to the same physical sample in both pipelines.
This module reproduces that exact RNG stream (same seed, same draw order --
molecule 1 then molecule 2, per realization) without modifying the spectrum
code. ``test_disorder.py`` asserts the two streams are bit-for-bit identical.
"""

from typing import Iterator, List, Tuple

import numpy as np

from .config import Config


def disorder_draws(cfg: Config, sigma: float, n_real: int) -> List[Tuple[float, float]]:
    """Return the list of ``(eps1_dis, eps2_dis)`` for ``n_real`` realizations.

    Identical to what :mod:`spectrum.spectrum` draws for the same ``sigma`` and
    ``cfg`` (seed, ``eps1``, ``eps2``). At ``sigma == 0`` every realization
    returns ``(cfg.eps1, cfg.eps2)`` unchanged.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    draws: List[Tuple[float, float]] = []
    for _ in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        draws.append((eps1_dis, eps2_dis))
    return draws


def iter_disorder(cfg: Config, sigma: float, n_real: int) -> Iterator[Tuple[int, float, float]]:
    """Yield ``(realization_index, eps1_dis, eps2_dis)`` in the spectrum's order."""
    for r, (eps1_dis, eps2_dis) in enumerate(disorder_draws(cfg, sigma, n_real)):
        yield r, eps1_dis, eps2_dis
