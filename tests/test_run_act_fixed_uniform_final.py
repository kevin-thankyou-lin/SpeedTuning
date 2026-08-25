import json

import pytest

from scripts import run_act_fixed_uniform_final as module


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def rollout(self, schedule, seed):
        self.calls.append((list(schedule), seed))
        return {"seed": seed, "schedule": list(schedule), "success": True}


def test_speed_slugs_are_stable():
    assert [module.speed_slug(speed) for speed in module.SPEEDS] == [
        "uniform_1p5x", "uniform_2p0x", "uniform_2p5x", "uniform_3p0x"
    ]


def test_evaluate_reuses_only_identity_matching_receipts(tmp_path):
    runtime = FakeRuntime()
    root = tmp_path / "run"
    cached = root / "uniform_2p0x" / "states" / "11.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps({
        "seed": 11,
        "schedule": [2.0, 2.0, 2.0, 2.0],
        "success": False,
    }))

    values = module.evaluate(runtime, root, [11, 12], 2.0)

    assert [value["seed"] for value in values] == [11, 12]
    assert runtime.calls == [([2.0, 2.0, 2.0, 2.0], 12)]


def test_evaluate_rejects_mismatched_cached_schedule(tmp_path):
    runtime = FakeRuntime()
    path = tmp_path / "uniform_2p0x" / "states" / "11.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"seed": 11, "schedule": [3.0] * 4}))

    with pytest.raises(RuntimeError, match="identity mismatch"):
        module.evaluate(runtime, tmp_path, [11], 2.0)

