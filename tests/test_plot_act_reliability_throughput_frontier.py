from pathlib import Path

from scripts import plot_act_reliability_throughput_frontier as module


def test_parse_results_reads_success_and_throughput(tmp_path: Path):
    path = tmp_path / "RESULTS.md"
    path.write_text(
        "| Task | Method | Success | SR | Speedup | Throughput | Safety | Physics |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
        "| Pick | Native 1x | 49/50 | 0.98 | 1.0x | +0.0% | 0 | 0 |\n"
        "| Tea | Native 1x | 50/50 | 1.00 | 1.0x | +0.0% | 0 | 0 |\n"
        "| Insertion | Native 1x | 48/50 | 0.96 | 1.0x | -2.5% | 0 | 0 |\n"
    )

    values = module.parse_results(path)

    assert values[0].success_rate_percent == 98.0
    assert values[2].throughput_delta == -2.5


def test_pareto_frontier_drops_jointly_worse_method():
    values = [
        module.Result("Pick", "a", 50, 50, 20.0),
        module.Result("Pick", "b", 45, 50, 40.0),
        module.Result("Pick", "c", 45, 50, 30.0),
        module.Result("Pick", "d", 40, 50, 50.0),
    ]

    frontier = module.pareto_frontier(values)

    assert [value.method for value in frontier] == ["d", "b", "a"]


def test_success_axis_expands_high_reliability_and_compresses_low_reliability():
    low_width = module.success_axis_position(80) - module.success_axis_position(0)
    high_width = module.success_axis_position(100) - module.success_axis_position(80)

    assert low_width == 15
    assert high_width == 85
    assert high_width > 5 * low_width
    assert module.success_axis_position(85) == 30
    assert module.success_axis_position(95) == 72.5


def test_sail_and_volt_have_explicit_plot_styles():
    assert module.method_style("SAIL-inspired")["color"] != "#b5b5b5"
    assert module.method_style("VOLT-style")["marker"] == "d"
