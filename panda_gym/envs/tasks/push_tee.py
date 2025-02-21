from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import *


class PushT(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.01,
        # distance_threshold=0.05,
        goal_xy_range=0.35,
        obj_xy_range=0.35,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.pos_weight = 1.0
        self.rot_weight = 1.0
        self.object_size = 0.02
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.get_ee_position = get_ee_position
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(
            length=1.5, width=1.0, height=0.4, x_offset=-0.3, lateral_friction=0.7
        )
        self.sim.create_tee(
            body_name="object",
            mass=2.0,
            lateral_friction=0.7,
        )
        self.sim.create_tee(
            body_name="target",
            mass=0.0,
            ghost=True,
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        object_orientation = np.array(
            self.sim.get_base_rotation("object", "quaternion")
        )
        ee_position = np.array(self.get_ee_position())
        return np.concatenate([object_position, object_orientation, ee_position])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_pose = self._sample_object()
        self.sim.set_base_pose("target", self.goal[0:3], self.goal[3:-3])
        self.sim.set_base_pose("object", object_pose[0:3], object_pose[3:])

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # goal = np.array(
        #     [0.0, 0.0, self.object_size / 2, 0.0, 0.0, 0.0, 0.0, 0, 0, 0]
        # )  # dimension 10, note that the last 3 are for the end-effector, which is just a place holder here
        goal = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0]
        )  # dimension 10, note that the last 3 are for the end-effector, which is just a place holder here
        pos_noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        rot_noise_euler = np.array(
            [0, 0, self.np_random.uniform(-np.pi, np.pi, size=1)[0]]
        )  # Euler
        rot_noise = euler_to_quaternion(rot_noise_euler)
        noise = np.concatenate([pos_noise, rot_noise, np.zeros(3)])
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        # object_position = np.array([0.0, 0.0, self.object_size / 2])
        object_position = np.array([0.0, 0.0, 0.0])
        pos_noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += pos_noise

        object_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        rot_noise_euler = np.array(
            [0, 0, self.np_random.uniform(-np.pi, np.pi, size=1)[0]]
        )  # Euler
        rot_noise = euler_to_quaternion(rot_noise_euler)
        object_orientation += rot_noise

        return np.concatenate([object_position, object_orientation])

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        d = self.pos_weight * distance(
            achieved_goal[0:3], desired_goal[0:3]
        ) + self.rot_weight * angle_distance(
            achieved_goal[3:-3], desired_goal[3:-3])
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        # Here we also want to add a reward function for robot arm to reach the object
        d_ee = (
            distance(achieved_goal[0:3], achieved_goal[-3:]).astype(np.float32)
            if achieved_goal.ndim == 1
            else distance(achieved_goal[:, 0:3], achieved_goal[:, -3:]).astype(
                np.float32
            )
        )
        # d_ee = 0.0 if d_ee < 0.15 else d_ee
        d_ee = np.where(d_ee < 0.08, 0.0, d_ee)
        # if d_ee < 0.08:
        #     import pdb; pdb.set_trace()
        d_obj = (
            self.pos_weight * distance(achieved_goal[0:3], desired_goal[0:3])
            + self.rot_weight * angle_distance(achieved_goal[3:-3], desired_goal[3:-3])
            if achieved_goal.ndim == 1
            else self.pos_weight * distance(achieved_goal[:, 0:3], desired_goal[:, 0:3])
            + self.rot_weight
            * angle_distance(achieved_goal[:, 3:-3], desired_goal[:, 3:-3])
        )

        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -(d_obj.astype(np.float32) + d_ee)
