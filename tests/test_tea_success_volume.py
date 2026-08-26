from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from sim_tasks import (
    _oriented_boxes_overlap,
    _point_in_oriented_box,
    tea_bag_overlaps_cup_volume,
    tea_bag_reward,
)


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
TEA_XMLS = (
    "bimanual_viperx_ee_transfer_tea_bag.xml",
    "bimanual_viperx_transfer_tea_bag.xml",
)


class _NamedValues(dict):
    def __getitem__(self, key):
        return np.asarray(super().__getitem__(key), dtype=np.float64)


def _physics_with_tea_bag(center, rotation=np.eye(3), half_extents=(0.02,) * 3):
    data = SimpleNamespace(contact=[], ncon=0)
    named_data = SimpleNamespace(
        geom_xpos=_NamedValues(tea_bag=center),
        geom_xmat=_NamedValues(tea_bag=rotation),
        site_xpos=_NamedValues(cup_success_volume=[-0.1, 0.6, 0.0425]),
        site_xmat=_NamedValues(cup_success_volume=np.eye(3)),
    )
    named_model = SimpleNamespace(
        geom_size=_NamedValues(tea_bag=half_extents),
        site_size=_NamedValues(cup_success_volume=[0.04, 0.04, 0.0375])
    )
    return SimpleNamespace(
        data=data,
        model=SimpleNamespace(id2name=lambda *_: None),
        named=SimpleNamespace(data=named_data, model=named_model),
    )


@pytest.mark.parametrize("xml_name", TEA_XMLS)
def test_tea_models_define_the_same_noncolliding_success_volume(xml_name):
    root = ElementTree.parse(ASSET_ROOT / xml_name).getroot()
    volume = root.find(".//body[@name='cup']/site[@name='cup_success_volume']")

    assert volume is not None
    assert volume.attrib["type"] == "box"
    assert volume.attrib["pos"] == "0 0 0.0425"
    assert volume.attrib["size"] == "0.04 0.04 0.0375"


def test_oriented_box_membership_uses_the_box_local_frame():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert _point_in_oriented_box(
        np.array([0.0, 0.03, 0.0]),
        np.zeros(3),
        rotation,
        np.array([0.04, 0.01, 0.01]),
    )
    assert not _point_in_oriented_box(
        np.array([0.03, 0.0, 0.0]),
        np.zeros(3),
        rotation,
        np.array([0.04, 0.01, 0.01]),
    )


def test_oriented_box_overlap_uses_all_fifteen_separating_axes():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert _oriented_boxes_overlap(
        np.zeros(3),
        np.eye(3),
        np.array([0.04, 0.01, 0.01]),
        np.array([0.0, 0.045, 0.0]),
        rotation,
        np.array([0.04, 0.01, 0.01]),
    )
    assert not _oriented_boxes_overlap(
        np.zeros(3),
        np.eye(3),
        np.array([0.04, 0.01, 0.01]),
        np.array([0.0, 0.061, 0.0]),
        rotation,
        np.array([0.04, 0.01, 0.01]),
    )


@pytest.mark.parametrize(
    "center",
    [
        [-0.1, 0.6, 0.005],
        [-0.1, 0.6, 0.04],
        [-0.06, 0.64, 0.08],
    ],
)
def test_tea_reward_succeeds_anywhere_inside_inclusive_cup_volume(center):
    assert tea_bag_reward(_physics_with_tea_bag(center)) == 3


@pytest.mark.parametrize(
    "center",
    [
        [-0.1, 0.6, 0.0999],
        [-0.1, 0.5401, 0.06],
        [-0.1599, 0.6, 0.06],
    ],
)
def test_tea_reward_rejects_overlap_only_when_center_is_outside(center):
    physics = _physics_with_tea_bag(center)
    assert tea_bag_overlaps_cup_volume(physics)
    assert tea_bag_reward(physics) == 0


@pytest.mark.parametrize(
    "center",
    [
        [-0.1, 0.6, 0.1001],
        [-0.1601, 0.6, 0.04],
        [-0.1, 0.6601, 0.04],
        [-0.1, 0.6, -0.0151],
    ],
)
def test_tea_reward_rejects_bag_volume_separated_from_cup_volume(center):
    assert tea_bag_reward(_physics_with_tea_bag(center)) == 0
