"""Four-state oracle phase observation for the bounded RL comparison."""

from __future__ import annotations

import numpy as np


PHASES = ("pre_grasp", "grasp_lift", "transport", "interaction")
TASK_OBJECTS = {"pick_and_place": 1, "tea_bag": 1, "insertion": 2}


class OraclePhaseEncoder:
    """Return only a latched one-hot phase derived from simulator object state."""

    requires_images = False
    output_dim = 4

    def __init__(self, task_name: str):
        if task_name not in TASK_OBJECTS:
            raise ValueError(f"unsupported task: {task_name}")
        self.task_name = task_name
        self.reset()

    def reset(self):
        self.phase_index = 0
        self.initial_objects = None

    def _features(self, observation):
        env_state = np.asarray(observation["env_state"], dtype=np.float64)
        objects = [
            env_state[index * 7:index * 7 + 3]
            for index in range(TASK_OBJECTS[self.task_name])
        ]
        if self.initial_objects is None:
            self.initial_objects = [item.copy() for item in objects]
        effectors = {
            "left": np.asarray(observation["mocap_pose_left"], dtype=np.float64)[:3],
            "right": np.asarray(observation["mocap_pose_right"], dtype=np.float64)[:3],
        }
        return objects, effectors

    def _advance(self, objects, effectors):
        if self.phase_index == 0:
            if self.task_name == "insertion":
                measure = max(
                    np.linalg.norm(objects[0] - effectors["right"]),
                    np.linalg.norm(objects[1] - effectors["left"]),
                )
            else:
                measure = np.linalg.norm(objects[0] - effectors["right"])
            if measure <= 0.08:
                self.phase_index = 1
        elif self.phase_index == 1:
            lifts = [
                current[2] - initial[2]
                for current, initial in zip(objects, self.initial_objects)
            ]
            measure = min(lifts) if self.task_name == "insertion" else lifts[0]
            if measure >= 0.03:
                self.phase_index = 2
        elif self.phase_index == 2:
            if self.task_name == "tea_bag":
                measure = np.linalg.norm(objects[0][:2] - np.asarray([-0.1, 0.6]))
                threshold = 0.05
            elif self.task_name == "insertion":
                measure = np.linalg.norm(objects[0] - objects[1])
                threshold = 0.1507661311
            else:
                measure = np.linalg.norm(objects[0] - effectors["left"])
                threshold = 0.05
            if measure <= threshold:
                self.phase_index = 3

    def __call__(self, observation):
        objects, effectors = self._features(observation)
        self._advance(objects, effectors)
        value = np.zeros(len(PHASES), dtype=np.float32)
        value[self.phase_index] = 1.0
        return value

    def decision_token(self):
        """Expose only phase identity so the environment can interrupt at a boundary."""
        return self.phase_index

    def spec(self):
        return {
            "type": "oracle_phase_one_hot",
            "task": self.task_name,
            "phases": list(PHASES),
            "boundary_set": "manual-oracle-four-segment-v1",
        }


def create_oracle_phase_encoder(task_name: str, **_kwargs):
    return OraclePhaseEncoder(task_name)
