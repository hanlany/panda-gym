import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import panda_gym.assets


class PyBullet:
    """Convenient class to use PyBullet physics engine.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
        background_color (np.ndarray, optional): The background color as (red, green, blue).
            Defaults to np.array([223, 54, 45]).
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        n_substeps: int = 20,
        background_color: Optional[np.ndarray] = None,
        renderer: str = "Tiny",
    ) -> None:
        self.render_mode = render_mode
        background_color = background_color if background_color is not None else np.array([223.0, 54.0, 45.0])
        self.background_color = background_color.astype(np.float32) / 255
        options = "--background_color_red={} --background_color_green={} --background_color_blue={}".format(
            *self.background_color
        )
        if self.render_mode == "human":
            self.connection_mode = p.GUI
        elif self.render_mode == "rgb_array":
            if renderer == "OpenGL":
                self.connection_mode = p.GUI
            elif renderer == "Tiny":
                # self.connection_mode = p.DIRECT
                self.connection_mode = None
            else:
                raise ValueError("The 'renderer' argument is must be in {'Tiny', 'OpenGL'}")
        else:
            raise ValueError("The 'render' argument is must be in {'rgb_array', 'human'}")
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        
        self.physics_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self) -> None:
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()
            
    def is_collided(self) -> bool:
        return len(self.physics_client.getContactPoints()) > 0
    
    def get_collision_distance(self, body1: str, body2: str) -> float:
        closest_points = self.physics_client.getClosestPoints(self._bodies_idx[body1], self._bodies_idx[body2], 0.1)
        if len(closest_points) == 0:
            return 0.1
        min_distance = 1000
        for point in closest_points:
            min_distance = min(min_distance, point[8])
        return np.round(min_distance, 6)

    def close(self) -> None:
        """Close the simulation."""
        if self.physics_client.isConnected():
            self.physics_client.disconnect()

    def save_state(self) -> int:
        """Save the current simulation state.

        Returns:
            int: A state id assigned by PyBullet, which is the first non-negative
            integer available for indexing.
        """
        return self.physics_client.saveState()

    def restore_state(self, state_id: int) -> None:
        """Restore a simulation state.

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.restoreState(state_id)

    def remove_state(self, state_id: int) -> None:
        """Remove a simulation state. This will make this state_id available again for returning in save_state().

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.removeState(state_id)

    def render(
        self,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Args:
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Roll of the camera. Defaults to 0.
            mode (str, optional): Deprecated: This argument is deprecated and will be removed in a future
                version. Use the render_mode argument of the constructor instead.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if self.render_mode == "rgb_array":
            target_position = target_position if target_position is not None else np.zeros(3)
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, rgba, _, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                shadow=True,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            # With Python3.10, pybullet return flat tuple instead of array. So we need to build create the array.
            rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            return rgba[..., :3]

    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)

    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as quaternion (x, y, z, w).
        """
        orientation = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[1]
        return np.array(orientation)

    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        quaternion = self.get_base_orientation(body)
        if type == "euler":
            rotation = self.physics_client.getEulerFromQuaternion(quaternion)
            return np.array(rotation)
        elif type == "quaternion":
            return np.array(quaternion)
        else:
            raise ValueError("""type must be "euler" or "quaternion".""")

    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)

    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.physics_client.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The velocity.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[1]

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if len(orientation) == 3:
            orientation = self.physics_client.getQuaternionFromEuler(orientation)
        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_base_velocity(self, body: str, pos_v: np.ndarray, ang_v: np.ndarray) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            pos_v (np.ndarray): The linear velocity, as (vx, vy, vz).
            ang_v (np.ndarray): The angular velocity, as (wx, wy, wz).
        """
        # if len(orientation) == 3:
        #     orientation = self.physics_client.getQuaternionFromEuler(orientation)
        # Assume that angular velocity is in euler angle
        self.physics_client.resetBaseVelocity(
            objectUniqueId=self._bodies_idx[body], linearVelocity=pos_v, angularVelocity=ang_v
        )

    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    def set_joint_angle(self, body: str, joint: int, angle: float, velocity: float = 0.0) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
            velocity (float): Target velocity.
        """
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle, targetVelocity=velocity)

    def set_joint_angles_with_vel(self, body: str, joints: np.ndarray, angles: np.ndarray, velocities: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
            velocities (np.ndarray): List of target velocities, as a list of floats.
        """
        for joint, angle, velocity in zip(joints, angles, velocities):
            self.set_joint_angle(body=body, joint=joint, angle=angle, velocity=velocity)
        
    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def inverse_kinematics(self, body: str, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        joint_state = self.physics_client.calculateInverseKinematics(
            bodyIndex=self._bodies_idx[body],
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
        )
        return np.array(joint_state)

    def place_visualizer(self, target_position: np.ndarray, distance: float, yaw: float, pitch: float) -> None:
        """Orient the camera used for rendering.

        Args:
            target (np.ndarray): Target position, as (x, y, z).
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        if texture is not None:
            texture_path = os.path.join(panda_gym.assets.get_data_path(), texture)
            texture_uid = self.physics_client.loadTexture(texture_path)
            self.physics_client.changeVisualShape(self._bodies_idx[body_name], -1, textureUniqueId=texture_uid)

    def create_tee(
        self,
        length: float = 0.1,
        width: float = 0.08,
        height: float = 0.02,
        thickness: float = 0.03,
        mass: float = 1.0,
        body_name: str = "tee",
        ghost: bool = False,
        x_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """ Create a T box.
        
        Args:
            length (float): The length of the l part of the T (x direction).
            width (float): The width the - part of the T (y direction)
            height (float): The height of letter tee.
            thickness (float): The thickness of the letter tee.
            mass (float): The mass in kg.
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            x_offset (float, optional): The offset in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = (np.array([0, 1, 0, 1]) if not ghost else np.array([0, 1, 0, 0.3]))
        specular_color = np.zeros(3)
        
        # Horizontal bar position
        hori_position = [[-thickness/2 , 0, height/2]]
        # Vertical bar position
        verti_position = [[length/2, 0, height/2]]

        # Create Visual Shape Id and Collision Shape Id
        visualShapeId = self.physics_client.createVisualShapeArray(
            shapeTypes=[self.physics_client.GEOM_BOX]*2,
            halfExtents=[[thickness/2, width/2, height/2], [length/2, thickness/2, height/2]],
            visualFramePositions=hori_position + verti_position
        )
        if not ghost:
            collisionShapeId = self.physics_client.createCollisionShapeArray(
                shapeTypes=[self.physics_client.GEOM_BOX]*2,
                halfExtents=[[thickness/2, width/2, height/2], [length/2, thickness/2, height/2]],
                collisionFramePositions=hori_position + verti_position
            )
            bodies_mass = mass
        else:
            collisionShapeId = -1
            bodies_mass = 0.0

        # Create MultiBody
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseMass=bodies_mass,
            baseVisualShapeIndex=visualShapeId,
            baseCollisionShapeIndex=collisionShapeId,
            basePosition=np.zeros(3),
        )
        self.physics_client.changeVisualShape(
            self._bodies_idx[body_name],
            -1,
            rgbaColor=rgba_color,
            specularColor=specular_color
        )
        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)
        
    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        position = position if position is not None else np.zeros(3)
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_plane(self, z_offset: float) -> None:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.15, 0.15, 0.15, 1.0]),
        )

    def create_table(
        self,
        length: float,
        width: float,
        height: float,
        x_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offset in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, 0.0, -height / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
        )
    
    def create_shelve(
        self,
        length: float,
        width: float,
        height: float,
        body_name: str = "shelve",
        layers: int = 3,
        x_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """ Create a fixed shelve. Bottom is z=0, centered in y.
        
        Args:
            length (float): The length of the shelve (x direction).
            width (float): The width of the shelve (y direction)
            height (float): The height of the shelve.
            layers (int): The number of layers of the shelve.
            x_offset (float, optional): The offset in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = np.array([0, 1, 0, 0.8])
        specular_color = np.zeros(3)
        
        # Poles frame position
        pole_position = []
        for i in range(4):
            pole_position.append([x_offset + (i%2) * length, width/2 * ( -1 if i < 2 else 1) , height/2])
        # Layer frame position
        layer_position = []
        for i in range(layers):
            layer_position.append([x_offset + length/2, 0.0, i*(height/(layers-1))])
        # Create Visual Shape Id and Collision Shape Id
        visualShapeId = self.physics_client.createVisualShapeArray(
            shapeTypes=[self.physics_client.GEOM_CYLINDER]*4 + [self.physics_client.GEOM_BOX]*layers,
            halfExtents=[[0, 0, 0]]*4 + [[length/2, width/2, 0.01]]*layers,
            radii=[0.02]*4 + [0.0]*layers,
            lengths=[height]*4 + [0]*layers,
            visualFramePositions=pole_position + layer_position
        )
        collisionShapeId = self.physics_client.createCollisionShapeArray(
            shapeTypes=[self.physics_client.GEOM_CYLINDER]*4 + [self.physics_client.GEOM_BOX]*layers,
            halfExtents=[[0, 0, 0]]*4 + [[length/2, width/2, 0.01]]*layers,
            radii=[0.02]*4 + [0.0]*layers,
            lengths=[height]*4 + [0]*layers,
            collisionFramePositions=pole_position + layer_position
        )
        # Create MultiBody
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visualShapeId,
            baseCollisionShapeIndex=collisionShapeId,
            basePosition=np.zeros(3),
        )
        self.physics_client.changeVisualShape(
            self._bodies_idx[body_name],
            -1,
            rgbaColor=rgba_color,
            specularColor=specular_color
        )
        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )
