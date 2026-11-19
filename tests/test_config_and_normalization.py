import torch
from tensor_compression.data.normalization import normalize_tensor, denormalize_tensor


def test_channelwise_minmax_normalizes_each_channel_independently() -> None:
    config = {"mode": "minmax", "scope": "channel", "clip_min": None, "clip_max": None}
    tensor = torch.tensor(
        [
            [[0.0, 2.0], [4.0, 6.0]],
            [[10.0, 14.0], [18.0, 22.0]],
        ]
    )

    normalized, _ = normalize_tensor(tensor, config)

    expected = torch.tensor(
        [
            [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
            [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
        ]
    )
    assert torch.allclose(normalized, expected, atol=1.0e-6)


def test_channelwise_zscore_roundtrips_with_denormalize() -> None:
    config = {"mode": "zscore", "scope": "channel", "clip_min": None, "clip_max": None}
    tensor = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[100.0, 120.0], [140.0, 160.0]],
        ]
    )

    normalized, state = normalize_tensor(tensor, config)
    restored = denormalize_tensor(normalized, state)

    assert torch.allclose(restored, tensor, atol=1.0e-5)
