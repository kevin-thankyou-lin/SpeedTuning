import json

import torch

from scripts.monitor_relative_training import _latest_training_step, _stable_snapshot


def test_latest_training_step_uses_log_then_completion(tmp_path):
    log = tmp_path / "train.log"
    complete = tmp_path / "training_complete.json"
    log.write_text(
        "not json\n"
        + json.dumps({"step": 500, "validation_loss": 2.0})
        + "\n"
        + json.dumps({"step": 1500, "validation_loss": 1.0})
        + "\n"
    )
    assert _latest_training_step(log, complete) == 1500
    complete.write_text(json.dumps({"steps": 20000}))
    assert _latest_training_step(log, complete) == 20000


def test_stable_snapshot_is_loadable_and_preserves_step(tmp_path):
    source = tmp_path / "best.pt"
    destination = tmp_path / "snapshots" / "best-step-00500.pt"
    torch.save({"step": 500, "weight": torch.arange(3)}, source)

    payload = _stable_snapshot(source, destination)

    assert payload["step"] == 500
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert loaded["step"] == 500
    assert torch.equal(loaded["weight"], torch.arange(3))
