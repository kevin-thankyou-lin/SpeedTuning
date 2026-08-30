import json
import subprocess
import sys
from pathlib import Path

from scripts.run_act_strider_uniform_ablation_v29 import UNIFORM, paired_summary


def record(seed, success, steps=100):
    return {
        "seed": seed,
        "success": success,
        "first_success_step": steps if success else None,
        "physics_steps": 400,
        "physics_error": None,
        "safety_violation": None,
    }


def test_contract_registers_only_missing_uniform_rollouts():
    repo = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (repo / "experiments/act_strider_uniform_ablation_v29/contract.json").read_text()
    )
    assert UNIFORM == [2.0] * 4
    assert contract["comparisons"]["pick"]["new_rollouts"] == 50
    assert contract["comparisons"]["tea"]["new_rollouts"] == 50
    assert contract["comparisons"]["insertion"]["new_rollouts"] == 0
    assert contract["comparisons"]["insertion"]["strider_schedule"] == [1.5] * 4
    assert contract["accounting"]["new_rollouts"] == 100


def test_paired_summary_preserves_discordant_outcomes():
    strider = [record(1, True, 80), record(2, True), record(3, False), record(4, False)]
    uniform = [record(1, True, 100), record(2, False), record(3, True), record(4, False)]
    result = paired_summary(strider, uniform)
    assert result["success_contingency"] == {
        "both_success": 1, "strider_only": 1, "uniform_only": 1, "both_fail": 1,
    }
    assert result["success_delta_strider_minus_uniform"] == 0
    assert result["both_success_step_pairs"][0]["uniform_over_strider_step_ratio"] == 1.25


def test_workflow_is_l40_and_runs_only_pick_and_tea():
    repo = Path(__file__).resolve().parents[1]
    workflow = (repo / "osmo/act_strider_uniform_ablation_v29_l40.yaml").read_text()
    assert "platform: ovx-l40" in workflow
    assert "for TASK in pick tea" in workflow
    assert "--v28-root \"$V28\"" in workflow
    assert "v20_v26_v27_v28_rollouts_reexecuted" in (
        repo / "scripts/finalize_act_strider_uniform_ablation_v29.py"
    ).read_text()


def test_entrypoints_work_from_arbitrary_cwd(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    for script in (
        "scripts/run_act_strider_uniform_ablation_v29.py",
        "scripts/finalize_act_strider_uniform_ablation_v29.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(repo / script), "--help"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
