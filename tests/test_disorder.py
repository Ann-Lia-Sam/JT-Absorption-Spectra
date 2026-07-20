"""The vibronic/heatmap workflow must reuse the spectrum's exact disorder draws.

:mod:`spectrum.disorder` reproduces the RNG stream that :mod:`spectrum.spectrum`
generates inline. This test asserts they are bit-for-bit identical so that a
given realization index refers to the same physical sample in both pipelines.
"""
import numpy as np

from spectrum.config import Config
from spectrum.disorder import disorder_draws


def _spectrum_inline_draws(cfg, sigma, n_real):
    """Replicate spectrum.py's inline draw loop exactly (the reference stream)."""
    rng = np.random.default_rng(cfg.rng_seed)
    out = []
    for _ in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        out.append((eps1_dis, eps2_dis))
    return out


def test_disorder_stream_matches_spectrum():
    cfg = Config()
    for sigma in (0.0, 0.05, 0.24):
        ref = _spectrum_inline_draws(cfg, sigma, 50)
        got = disorder_draws(cfg, sigma, 50)
        assert len(ref) == len(got)
        for (e1r, e2r), (e1g, e2g) in zip(ref, got):
            assert e1r == e1g and e2r == e2g


def test_disorder_zero_sigma_is_unshifted():
    cfg = Config()
    draws = disorder_draws(cfg, 0.0, 5)
    for e1, e2 in draws:
        assert e1 == cfg.eps1 and e2 == cfg.eps2


if __name__ == "__main__":
    test_disorder_stream_matches_spectrum()
    test_disorder_zero_sigma_is_unshifted()
    print("disorder tests passed")
