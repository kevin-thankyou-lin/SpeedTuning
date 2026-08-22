import pytest


def test_masked_l1_is_invariant_to_padded_suffix_length():
    torch = pytest.importorskip("torch")
    from policy import masked_l1_loss

    target = torch.ones((2, 48, 14))
    prediction = torch.zeros_like(target)
    is_pad = torch.ones((2, 48), dtype=torch.bool)
    is_pad[:, :8] = False
    assert masked_l1_loss(target, prediction, is_pad).item() == pytest.approx(1.0)
