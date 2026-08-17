import numpy as np

from ee_sim_env import make_ee_sim_env
from scripted_policy import InsertionPolicy


def test_pose_solver_preserves_nominal_object_path():
    policy = InsertionPolicy()
    env = make_ee_sim_env("insertion", render_images=False, seed=7)
    first = env.reset()
    policy.generate_trajectory(first)

    nominal_relation = policy._nominal_object_in_gripper["right"]
    actual_relation = nominal_relation.copy()
    actual_relation[:3] += np.array([0.004, -0.003, 0.002])
    actual_relation[3:] = np.array([0.92387953, 0.0, 0.38268343, 0.0])

    for waypoint in policy._nominal_suffix["right"]:
        adapted = policy._adapt_waypoint(
            waypoint, nominal_relation, actual_relation
        )
        nominal_gripper = np.concatenate([waypoint["xyz"], waypoint["quat"]])
        adapted_gripper = np.concatenate([adapted["xyz"], adapted["quat"]])
        desired_object = policy._compose_pose(nominal_gripper, nominal_relation)
        corrected_object = policy._compose_pose(adapted_gripper, actual_relation)
        np.testing.assert_allclose(corrected_object[:3], desired_object[:3], atol=1e-10)
        np.testing.assert_allclose(adapted["quat"], waypoint["quat"], atol=1e-10)


def test_insertion_replans_once_after_both_objects_are_lifted():
    env = make_ee_sim_env("insertion", render_images=False, seed=0)
    timestep = env.reset()
    policy = InsertionPolicy()
    rewards = []

    for _ in range(400):
        timestep = env.step(policy(timestep))
        rewards.append(int(timestep.reward or 0))

    assert max(rewards) == env.task.max_reward
    assert policy.replan_count == 1
    assert policy.replan_event["reward"] >= 2
    assert policy.replan_event["state_source"] == "privileged_sim_object_pose"
    assert policy.replan_event["correction_mode"].startswith("translation_only")
    assert 220 <= policy.replan_event["policy_time"] <= 284


def test_replan_can_be_disabled_for_action_identical_open_loop_control():
    env = make_ee_sim_env("insertion", render_images=False, seed=0)
    timestep = env.reset()
    policy = InsertionPolicy(enable_postgrasp_replan=False)

    for _ in range(400):
        timestep = env.step(policy(timestep))

    assert policy.replan_count == 0
    assert policy.replan_event is None
