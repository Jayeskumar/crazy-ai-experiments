import numpy as np
import torch

from src.model_b import WeightAE, WeightNormalizer, PCAModel


def test_ae_shapes_and_masking():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((10, 50)).astype(np.float32)
    sl = [slice(0, 20), slice(20, 50)]
    ae = WeightAE(50, 3, hidden=16, normalizer=WeightNormalizer(W, sl))
    assert ae.encode(W).shape == (10, 3)
    assert ae.reconstruct(W).shape == W.shape
    assert ae.reconstruct(W[0], mask_slices=[sl[0]]).shape == (50,)


def test_pca_masked_recovery_exact_in_span():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((3, 60))
    W = (rng.standard_normal((20, 3)) @ basis).astype(np.float64)
    sl = [slice(0, 30), slice(30, 60)]
    pca = PCAModel(W[:19], 3, sl)
    rec = pca.reconstruct(W[19], mask_slices=[sl[0]])
    assert np.abs(rec - W[19]).max() < 1e-4 * np.abs(W[19]).max() + 1e-6
