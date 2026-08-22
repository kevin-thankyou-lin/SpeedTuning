from pathlib import Path

from scripts.monitor_original_act_training import _ready


def test_checkpoint_is_released_only_after_successor_or_completion(tmp_path):
    checkpoint = tmp_path / "policy_epoch_500_seed_0.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    assert not _ready(tmp_path, 500)
    (tmp_path / "policy_epoch_600_seed_0.ckpt").write_bytes(b"next")
    assert _ready(tmp_path, 500)

    final = tmp_path / "policy_epoch_1900_seed_0.ckpt"
    final.write_bytes(b"final")
    assert not _ready(tmp_path, 1900)
    (tmp_path / "training_complete.json").write_text("{}")
    assert _ready(tmp_path, 1900)
