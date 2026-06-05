import torch

from clsl_v2.losses.label_marginalization import LabelMarginalizationLoss
from clsl_v2.losses.lattice_consistency import ConditionalLatticeConsistency, evidence_lattice_consistency


def test_label_marginalization_reduces_to_masked_bce_when_no_edges():
    torch.manual_seed(0)
    b, k = 5, 6
    unary = torch.randn(b, k)
    pairwise = torch.zeros(b, k, k)
    labels = torch.randint(0, 2, (b, k)).float()
    mask = torch.randint(0, 2, (b, k)).float()
    loss_fn = LabelMarginalizationLoss(k)
    exact = loss_fn(unary, pairwise, labels, mask)
    bce = loss_fn.masked_bce_equivalent(unary, labels, mask)
    assert torch.allclose(exact, bce, atol=1e-5), (exact, bce)


def test_lattice_losses_are_finite():
    torch.manual_seed(1)
    b, k, d, f = 8, 6, 16, 20
    cp = torch.rand(b, k) * 0.8 + 0.1
    fp = torch.rand(b, k) * 0.8 + 0.1
    ce = torch.rand(b)
    fe = torch.rand(b)
    cm = torch.randint(0, 2, (b, f)).float()
    fm = torch.ones(b, f)
    base, mono = evidence_lattice_consistency(cp, fp, ce, fe, coarse_mask=cm, fine_mask=fm)
    assert torch.isfinite(base + mono)
    cond = ConditionalLatticeConsistency()
    loss, diag = cond(cp, fp, ce, fe, coarse_repr=torch.randn(b, d), coarse_mask=cm, fine_mask=fm, return_diagnostics=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(diag.consistency)
