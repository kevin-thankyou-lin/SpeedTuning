import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from original_act import episode_paths, split_original_act_episodes
from original_diffusion import (
    JointRangeNormalizer,
    ModelEMA,
    MultiviewObservationEncoder,
    OriginalDiffusionDataset,
    OriginalDiffusionPolicy,
)
from scripts.evaluate_original_diffusion import checkpoint_identity
from scripts.smoke_original_diffusion import _matched_smoke_contract, _successful_episode
from scripts.train_original_diffusion import (
    _evaluate_milestone,
    cosine_warmup_multiplier,
    deterministic_validation,
)


CAMERAS = ("angle", "left_wrist", "right_wrist")


def _episode(
    path,
    length=20,
    camera_names=CAMERAS,
    offset=0,
    *,
    seed=0,
    source_success=True,
    replay_success=True,
    task="pick_and_place",
):
    qpos = np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 100 + offset
    action = qpos + np.arange(14, dtype=np.float32) / 10 + 0.25
    with h5py.File(path, "w") as root:
        root.attrs.update(
            seed=seed,
            source_success=source_success,
            replay_success=replay_success,
            task=task,
        )
        root.create_dataset("observations/qpos", data=qpos)
        root.create_dataset("action", data=action)
        root.create_dataset("object_pose", data=np.arange(7, dtype=np.float64) + seed)
        images = root.create_group("observations/images")
        for camera_index, name in enumerate(camera_names):
            images.create_dataset(
                name,
                data=np.full((length, 12, 16, 3), 10 * camera_index + np.arange(length)[:, None, None, None], dtype=np.uint8),
            )
    return qpos, action


def _small_policy():
    return OriginalDiffusionPolicy(
        {
            "pretrained_backbone": False,
            "image_size": [32, 32],
            "down_dims": [32, 64, 128],
            "num_train_timesteps": 4,
            "num_inference_steps": 2,
        }
    )


def test_global_joint_range_is_train_only_and_round_trips(tmp_path):
    first = tmp_path / "episode_0.hdf5"
    second = tmp_path / "episode_1.hdf5"
    _, action = _episode(first)
    _episode(second, offset=100)
    normalizer = JointRangeNormalizer.fit([first])
    assert normalizer.action_low.shape == (14,)
    assert normalizer.action_high.shape == (14,)
    assert normalizer.action_high.max() < 100
    normalized = normalizer.normalize_action(action)
    np.testing.assert_allclose(normalizer.denormalize_action(normalized), action, atol=2e-6)
    restored = JointRangeNormalizer.from_state_dict(normalizer.state_dict())
    np.testing.assert_array_equal(restored.qpos_low, normalizer.qpos_low)


def test_dataset_alignment_padding_progress_and_camera_order(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    qpos, action = _episode(path)
    normalizer = JointRangeNormalizer.fit([path])
    dataset = OriginalDiffusionDataset([path], normalizer, image_size=(12, 16))
    image, observed, target, is_pad = dataset[5]
    assert image.shape == (2, 3, 3, 12, 16)
    assert observed.shape == (2, 15)
    assert target.shape == (16, 14)
    np.testing.assert_allclose(
        normalizer.denormalize_action(target.numpy())[1], action[5], atol=2e-6
    )
    assert image[1, 0, 0, 0, 0] < image[1, 1, 0, 0, 0] < image[1, 2, 0, 0, 0]
    assert observed[1, -1].item() == pytest.approx(2 * 5 / 19 - 1)
    _, _, last_target, last_pad = dataset[19]
    np.testing.assert_allclose(
        normalizer.denormalize_action(last_target.numpy())[1], action[19], atol=2e-6
    )
    assert last_pad.tolist() == [False, False] + [True] * 14


def test_dataset_fails_closed_on_camera_mismatch(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    _episode(path, camera_names=("angle",))
    normalizer = JointRangeNormalizer.fit([path])
    with pytest.raises(ValueError, match="camera contract mismatch"):
        OriginalDiffusionDataset([path], normalizer)


def test_dataset_stride_selects_only_executable_replan_states(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    _, action = _episode(path, length=20)
    normalizer = JointRangeNormalizer.fit([path])
    dataset = OriginalDiffusionDataset(
        [path], normalizer, image_size=(12, 16), sample_stride=8
    )
    assert dataset.indices == [(0, 0), (0, 8), (0, 16)]
    _, _, target, is_pad = dataset[1]
    np.testing.assert_allclose(
        normalizer.denormalize_action(target.numpy())[1], action[8], atol=2e-6
    )
    assert not is_pad[1]
    with pytest.raises(ValueError, match="sample_stride must be positive"):
        OriginalDiffusionDataset([path], normalizer, sample_stride=0)


@pytest.mark.learned
def test_policy_fails_closed_on_training_execution_stride_mismatch():
    with pytest.raises(ValueError, match="training_sample_stride must match action_horizon"):
        OriginalDiffusionPolicy(
            {
                "pretrained_backbone": False,
                "action_horizon": 8,
                "training_sample_stride": 4,
            }
        )


def test_smoke_episode_is_selected_only_from_training_partition(tmp_path):
    paths = []
    for index in range(270):
        path = tmp_path / f"episode_{index}.hdf5"
        _episode(
            path,
            length=2,
            seed=1000 + index,
            source_success=False,
            replay_success=False,
        )
        paths.append(path)
    paths = episode_paths(tmp_path)
    train_paths, validation_paths = split_original_act_episodes(
        paths, seed=1, validation_count=20
    )
    validation_success = validation_paths[0]
    with h5py.File(validation_success, "r+") as root:
        root.attrs["source_success"] = True
        root.attrs["replay_success"] = True
    with pytest.raises(ValueError, match="no successful"):
        _successful_episode(train_paths)

    training_success = train_paths[-1]
    with h5py.File(training_success, "r+") as root:
        root.attrs["source_success"] = True
        root.attrs["replay_success"] = True
    all_paths, actual_train, actual_validation, episode = _matched_smoke_contract(tmp_path)
    assert all_paths == paths
    assert actual_train == train_paths
    assert actual_validation == validation_paths
    assert episode[0] == training_success
    assert episode[1] == 1000 + int(training_success.stem.removeprefix("episode_"))
    np.testing.assert_array_equal(
        episode[3], np.arange(7, dtype=np.float64) + episode[1]
    )


@pytest.mark.learned
def test_camera_fusion_preserves_identity_and_order():
    encoder = MultiviewObservationEncoder(pretrained=False, image_size=(32, 32)).eval()
    image = torch.zeros(1, 2, 3, 3, 32, 32)
    image[:, :, 0] = 0.1
    image[:, :, 1] = 0.5
    image[:, :, 2] = 0.9
    qpos = torch.zeros(1, 2, 15)
    with torch.inference_mode():
        first = encoder(image, qpos)
        swapped = encoder(image[:, :, [2, 1, 0]], qpos)
    assert first.shape == (1, 2 * (3 * 512 + 128))
    assert not torch.equal(first, swapped)


@pytest.mark.learned
def test_policy_loss_sampling_slice_and_fixed_validation_are_deterministic():
    torch.manual_seed(0)
    policy = _small_policy().eval()
    image = torch.rand(2, 2, 3, 3, 32, 32)
    qpos = torch.rand(2, 2, 15)
    action = torch.rand(2, 16, 14) * 2 - 1
    is_pad = torch.zeros(2, 16, dtype=torch.bool)
    generator = torch.Generator().manual_seed(5)
    first = policy.compute_loss(qpos, image, action, is_pad, generator=generator)
    generator = torch.Generator().manual_seed(5)
    second = policy.compute_loss(qpos, image, action, is_pad, generator=generator)
    assert torch.isfinite(first)
    assert torch.equal(first, second)
    with torch.inference_mode():
        sample = policy.sample(qpos[:1], image[:1], generator=torch.Generator().manual_seed(8))
    assert sample.shape == (1, 16, 14)
    assert policy.executed_slice(sample).shape == (1, 8, 14)
    assert torch.equal(policy.executed_slice(sample), sample[:, 1:9])
    loader = DataLoader(list(zip(image, qpos, action, is_pad)), batch_size=1)
    assert deterministic_validation(policy, loader, torch.device("cpu"), 11) == deterministic_validation(
        policy, loader, torch.device("cpu"), 11
    )


@pytest.mark.learned
def test_ema_updates_and_round_trips():
    model = torch.nn.Linear(3, 2)
    ema = ModelEMA(model)
    with torch.no_grad():
        model.weight.add_(1)
    ema.update(model)
    assert torch.equal(ema.model.weight, model.weight)
    state = ema.state_dict()
    restored = ModelEMA(torch.nn.Linear(3, 2))
    restored.load_state_dict(state)
    assert torch.equal(restored.model.weight, ema.model.weight)
    assert restored.optimization_step == 1


def test_cosine_warmup_and_checkpoint_identity(tmp_path):
    assert cosine_warmup_multiplier(0, 100, 10) == pytest.approx(0.1)
    assert cosine_warmup_multiplier(9, 100, 10) == pytest.approx(1)
    assert cosine_warmup_multiplier(100, 100, 10) == pytest.approx(0)
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"sealed")
    initial = checkpoint_identity(path)
    assert initial == checkpoint_identity(path)
    path.write_bytes(b"changed")
    assert initial != checkpoint_identity(path)


def test_milestone_evaluation_preserves_exact_seed_order(monkeypatch, tmp_path):
    observed = []

    def fake_rollout(task, model, normalizer, device, seed, camera_names):
        observed.append((task, seed, camera_names))
        return {"seed": seed, "success": seed % 2 == 0, "steps": 7, "max_reward": 4}

    monkeypatch.setattr("scripts.train_original_diffusion.rollout", fake_rollout)

    class Model:
        action_horizon = 8
        prediction_horizon = 16
        observation_horizon = 2

        def eval(self):
            return self

    result = _evaluate_milestone(
        "pick_and_place",
        Model(),
        object(),
        torch.device("cpu"),
        update=25_000,
        episodes=3,
        seed_base=9_400_000,
        camera_names=CAMERAS,
        output_dir=tmp_path,
    )
    assert observed == [
        ("pick_and_place", 9_400_000 + index, CAMERAS) for index in range(3)
    ]
    assert result["successes"] == 2
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["replan_interval"] == 8
    assert (tmp_path / "update-025000.json").exists()
    assert not (tmp_path / "update-025000.json.partial").exists()
