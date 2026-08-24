import torch

from detr.models.position_encoding import PositionEmbeddingSine


def test_dense_coordinate_grid_matches_retained_cumsum_exactly():
    values = torch.zeros((2, 3, 11, 13), dtype=torch.float32)
    encoder = PositionEmbeddingSine(num_pos_feats=16, normalize=True)

    actual = encoder(values)

    ones = torch.ones_like(values[:, [0]])[:, 0]
    y_embed = ones.cumsum(1, dtype=torch.float32)
    x_embed = ones.cumsum(2, dtype=torch.float32)
    epsilon = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * encoder.scale
    x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * encoder.scale
    dim_t = torch.arange(16, dtype=torch.float32)
    dim_t = encoder.temperature ** (2 * (dim_t // 2) / 16)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    expected = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dense_position_encoding_allows_strict_cuda_determinism():
    if not torch.cuda.is_available():
        return
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values = torch.zeros((1, 3, 11, 13), dtype=torch.float32, device="cuda")
        result = PositionEmbeddingSine(num_pos_feats=16, normalize=True)(values)
        assert torch.all(torch.isfinite(result))
    finally:
        torch.use_deterministic_algorithms(previous)
