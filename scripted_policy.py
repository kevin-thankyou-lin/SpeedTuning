import numpy as np
from pyquaternion import Quaternion

from sim_tasks import normalize_task_name


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def reset(self):
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    def maybe_replan(self, ts):
        """Update a generated trajectory from a causal observation, if supported."""

        del ts

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"] + 1e-8)
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts, step_inc=1):
        if step_inc <= 0:
            raise ValueError("step_inc must be positive")

        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        self.maybe_replan(ts)

        curr_left_waypoint, next_left_waypoint = self._waypoint_pair(
            self.left_trajectory, self.step_count
        )
        curr_right_waypoint, next_right_waypoint = self._waypoint_pair(
            self.right_trajectory, self.step_count
        )

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(
            curr_left_waypoint, next_left_waypoint, self.step_count
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += step_inc
        return np.concatenate([action_left, action_right])

    @staticmethod
    def _waypoint_pair(trajectory, timestep):
        """Find a safe interpolation bracket, including at the final waypoint."""

        if not trajectory or len(trajectory) < 2:
            raise ValueError("A scripted trajectory must contain at least two waypoints")
        if timestep >= trajectory[-1]["t"]:
            return trajectory[-1], trajectory[-1]
        for current, following in zip(trajectory[:-1], trajectory[1:]):
            if current["t"] <= timestep < following["t"]:
                return current, following
        return trajectory[0], trajectory[1]


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0., 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]

class PickAndTransferTeaBagPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])

        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        #print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat_org = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat_org * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90)
        gripper_move_quat = gripper_pick_quat_org * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-20)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([-0.1, 0.6, 0.30])


        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 500, "xyz": init_mocap_pose_left[:3], "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        '''
        # Old policy, oscillation is a problem
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 70, "xyz": box_xyz + np.array([0.005, 0, -0.03]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 100, "xyz": box_xyz + np.array([0.005, 0, -0.03]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 170, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_move_quat.elements, "gripper": 0}, # approach meet position
            {"t": 200, "xyz": meet_xyz, "quat": gripper_move_quat.elements, "gripper": 0}, # move to meet position
            {"t": 420, "xyz": meet_xyz, "quat": gripper_move_quat.elements, "gripper": 0},  # open gripper
            {"t": 460, "xyz": meet_xyz, "quat": gripper_move_quat.elements, "gripper": 1}, # open gripper
            {"t": 500, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
        ]
        '''

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 70, "xyz": box_xyz + np.array([0.005, 0, -0.03]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 100, "xyz": box_xyz + np.array([0.005, 0, -0.03]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 150, "xyz": box_xyz + np.array([0.1, 0, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0}, # vertical align
            {"t": 250, "xyz": box_xyz + np.array([0.1, 0, 0.3]), "quat": gripper_move_quat.elements, "gripper": 0}, # rise up
            {"t": 400, "xyz": meet_xyz, "quat": gripper_move_quat.elements, "gripper": 0}, # move to meet position
            {"t": 420, "xyz": meet_xyz, "quat": gripper_move_quat.elements, "gripper": 0},  # open gripper
            {"t": 450, "xyz": meet_xyz, "quat": gripper_move_quat.elements, "gripper": 1}, # open gripper
            {"t": 500, "xyz": init_mocap_pose_right[:3], "quat": gripper_move_quat.elements, "gripper": 1},
        ]

class InsertionPolicy(BasePolicy):

    REPLAN_REWARD = 2
    REPLAN_MIN_POLICY_TIME = 220
    REPLAN_DEADLINE_POLICY_TIME = 284

    def __init__(self, inject_noise=False, enable_postgrasp_replan=True):
        super().__init__(inject_noise=inject_noise)
        self.enable_postgrasp_replan = bool(enable_postgrasp_replan)
        self.replan_count = 0
        self.replan_event = None
        self._nominal_object_in_gripper = None
        self._nominal_suffix = None

    def reset(self):
        super().reset()
        self.replan_count = 0
        self.replan_event = None
        self._nominal_object_in_gripper = None
        self._nominal_suffix = None

    @staticmethod
    def _relative_pose(parent_pose, child_pose):
        """Return the child pose expressed in the parent frame."""

        parent_pose = np.asarray(parent_pose, dtype=np.float64)
        child_pose = np.asarray(child_pose, dtype=np.float64)
        parent_quat = Quaternion(parent_pose[3:]).normalised
        child_quat = Quaternion(child_pose[3:]).normalised
        relative_quat = parent_quat.inverse * child_quat
        relative_xyz = parent_quat.inverse.rotate(child_pose[:3] - parent_pose[:3])
        return np.concatenate([relative_xyz, relative_quat.elements])

    @staticmethod
    def _compose_pose(parent_pose, child_in_parent):
        """Compose a world parent pose with a child pose in that frame."""

        parent_pose = np.asarray(parent_pose, dtype=np.float64)
        child_in_parent = np.asarray(child_in_parent, dtype=np.float64)
        parent_quat = Quaternion(parent_pose[3:]).normalised
        child_quat = Quaternion(child_in_parent[3:]).normalised
        child_xyz = parent_pose[:3] + parent_quat.rotate(child_in_parent[:3])
        child_world_quat = parent_quat * child_quat
        return np.concatenate([child_xyz, child_world_quat.elements])

    @staticmethod
    def _parent_pose_for_child(child_world_pose, child_in_parent):
        """Solve the parent pose that places a grasped child at a world pose."""

        child_world_pose = np.asarray(child_world_pose, dtype=np.float64)
        child_in_parent = np.asarray(child_in_parent, dtype=np.float64)
        child_world_quat = Quaternion(child_world_pose[3:]).normalised
        child_relative_quat = Quaternion(child_in_parent[3:]).normalised
        parent_quat = child_world_quat * child_relative_quat.inverse
        parent_xyz = child_world_pose[:3] - parent_quat.rotate(child_in_parent[:3])
        return np.concatenate([parent_xyz, parent_quat.elements])

    @staticmethod
    def _pose_error(reference_pose, observed_pose):
        reference_pose = np.asarray(reference_pose, dtype=np.float64)
        observed_pose = np.asarray(observed_pose, dtype=np.float64)
        reference_quat = Quaternion(reference_pose[3:]).normalised
        observed_quat = Quaternion(observed_pose[3:]).normalised
        rotation = reference_quat.inverse * observed_quat
        angle_degrees = np.degrees(abs(rotation.angle))
        if angle_degrees > 180.0:
            angle_degrees = 360.0 - angle_degrees
        return {
            "translation_m": float(np.linalg.norm(reference_pose[:3] - observed_pose[:3])),
            "rotation_deg": float(angle_degrees),
        }

    def _adapt_waypoint(self, waypoint, nominal_object_in_gripper, actual_object_in_gripper):
        nominal_gripper_pose = np.concatenate([waypoint["xyz"], waypoint["quat"]])
        desired_object_pose = self._compose_pose(
            nominal_gripper_pose, nominal_object_in_gripper
        )
        # Preserve the demonstrated gripper orientation.  The socket's free-joint
        # quaternion can jump between equivalent symmetric orientations while it
        # is grasped; following that quaternion caused a destructive wrist turn.
        # The achieved object offset in the gripper frame is still a reliable
        # translation correction.
        nominal_gripper_quat = Quaternion(waypoint["quat"]).normalised
        adapted_xyz = desired_object_pose[:3] - nominal_gripper_quat.rotate(
            actual_object_in_gripper[:3]
        )
        return {
            **waypoint,
            "xyz": adapted_xyz,
            "quat": np.asarray(waypoint["quat"], dtype=np.float64).copy(),
        }

    def maybe_replan(self, ts):
        if not self.enable_postgrasp_replan:
            return
        if self.replan_count:
            return
        reward = int(ts.reward or 0)
        if reward < self.REPLAN_REWARD:
            return
        if not self.REPLAN_MIN_POLICY_TIME <= self.step_count <= self.REPLAN_DEADLINE_POLICY_TIME:
            return

        env_state = np.asarray(ts.observation["env_state"], dtype=np.float64)
        current_objects = {
            "left": env_state[7:14],
            "right": env_state[:7],
        }
        current_grippers = {
            "left": np.asarray(ts.observation["mocap_pose_left"], dtype=np.float64),
            "right": np.asarray(ts.observation["mocap_pose_right"], dtype=np.float64),
        }
        actual_object_in_gripper = {
            side: self._relative_pose(current_grippers[side], current_objects[side])
            for side in ("left", "right")
        }

        for side in ("left", "right"):
            nominal_suffix = self._nominal_suffix[side]
            adapted_suffix = [
                self._adapt_waypoint(
                    waypoint,
                    self._nominal_object_in_gripper[side],
                    actual_object_in_gripper[side],
                )
                for waypoint in nominal_suffix
                if waypoint["t"] > self.step_count
            ]
            current = {
                "t": self.step_count,
                "xyz": current_grippers[side][:3].copy(),
                "quat": current_grippers[side][3:].copy(),
                "gripper": 0,
            }
            if not adapted_suffix:
                continue
            setattr(self, f"{side}_trajectory", [current, *adapted_suffix])

        self.replan_count = 1
        self.replan_event = {
            "policy_time": float(self.step_count),
            "reward": reward,
            "state_source": "privileged_sim_object_pose",
            "correction_mode": "translation_only_preserve_demonstrated_orientation",
            "grasp_error": {
                side: self._pose_error(
                    self._nominal_object_in_gripper[side],
                    actual_object_in_gripper[side],
                )
                for side in ("left", "right")
            },
        }

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]

        left_grasp_pose = np.concatenate(
            [self.left_trajectory[3]["xyz"], self.left_trajectory[3]["quat"]]
        )
        right_grasp_pose = np.concatenate(
            [self.right_trajectory[3]["xyz"], self.right_trajectory[3]["quat"]]
        )
        self._nominal_object_in_gripper = {
            "left": self._relative_pose(left_grasp_pose, socket_info),
            "right": self._relative_pose(right_grasp_pose, peg_info),
        }
        self._nominal_suffix = {
            "left": [{**waypoint} for waypoint in self.left_trajectory[4:]],
            "right": [{**waypoint} for waypoint in self.right_trajectory[4:]],
        }


POLICY_CLASSES = {
    "pick_and_place": PickAndTransferPolicy,
    "insertion": InsertionPolicy,
    "tea_bag": PickAndTransferTeaBagPolicy,
}


def make_scripted_policy(task_name, inject_noise=False):
    """Construct the matching historical scripted policy for a sim task."""

    return POLICY_CLASSES[normalize_task_name(task_name)](inject_noise=inject_noise)
