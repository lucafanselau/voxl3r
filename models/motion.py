import torch
from dataclasses import dataclass
from typing import Optional
from jaxtyping import Float
from torch import Tensor
import roma


@dataclass
class RigidBodyMotion:
    """Represents a rigid body motion with multiple possible representations"""

    rotation_matrix: Optional[Float[Tensor, "3 3"]] = None
    translation: Optional[Float[Tensor, "3"]] = None
    quaternion: Optional[Float[Tensor, "4"]] = None
    homogeneous_matrix: Optional[Float[Tensor, "4 4"]] = None

    def __post_init__(self):
        # Ensure at least one representation is provided
        if all(
            x is None
            for x in [self.rotation_matrix, self.quaternion, self.homogeneous_matrix]
        ):
            raise ValueError("At least one motion representation must be provided")

        # Convert to the other representations if not provided
        if self.homogeneous_matrix is not None:
            if self.rotation_matrix is None:
                self.rotation_matrix = self.homogeneous_matrix[:3, :3]
            if self.translation is None:
                self.translation = self.homogeneous_matrix[:3, 3]
            if self.quaternion is None:
                self.quaternion = roma.rotmat_to_unitquat(self.rotation_matrix)

        elif self.rotation_matrix is not None:
            if self.translation is None:
                self.translation = torch.zeros(3, device=self.rotation_matrix.device)
            if self.quaternion is None:
                self.quaternion = roma.rotmat_to_unitquat(self.rotation_matrix)
            if self.homogeneous_matrix is None:
                self.homogeneous_matrix = self._create_homogeneous_matrix()

        elif self.quaternion is not None:
            if self.translation is None:
                self.translation = torch.zeros(3, device=self.quaternion.device)
            if self.rotation_matrix is None:
                self.rotation_matrix = roma.unitquat_to_rotmat(self.quaternion)
            if self.homogeneous_matrix is None:
                self.homogeneous_matrix = self._create_homogeneous_matrix()

        # Validate rotation matrix
        if not roma.is_rotation_matrix(self.rotation_matrix, epsilon=1e-5):
            # Project to nearest rotation matrix using special Procrustes
            self.rotation_matrix = roma.special_procrustes(self.rotation_matrix)
            # Update other representations
            self.quaternion = roma.rotmat_to_unitquat(self.rotation_matrix)
            self.homogeneous_matrix = self._create_homogeneous_matrix()

    def _create_homogeneous_matrix(self) -> Float[Tensor, "4 4"]:
        """Creates a 4x4 homogeneous transformation matrix from rotation and translation"""
        # Create rigid transformation using RoMa
        rigid_transform = roma.Rigid(
            linear=self.rotation_matrix, translation=self.translation
        )
        return rigid_transform.to_homogeneous()

    def compose(self, other: "RigidBodyMotion") -> "RigidBodyMotion":
        """Composes this transformation with another"""
        # Use RoMa's composition of rigid transformations
        t1 = roma.Rigid(self.rotation_matrix, self.translation)
        t2 = roma.Rigid(other.rotation_matrix, other.translation)
        result = t1.compose(t2)

        return RigidBodyMotion(
            rotation_matrix=result.linear, translation=result.translation
        )

    def inverse(self) -> "RigidBodyMotion":
        """Returns the inverse transformation"""
        # Use RoMa's inverse of rigid transformation
        transform = roma.Rigid(self.rotation_matrix, self.translation)
        inv_transform = transform.inverse()

        return RigidBodyMotion(
            rotation_matrix=inv_transform.linear, translation=inv_transform.translation
        )

    def transform_points(
        self, points: Float[Tensor, "... 3"]
    ) -> Float[Tensor, "... 3"]:
        """Transforms a set of points using this rigid body motion"""
        transform = roma.Rigid(self.rotation_matrix, self.translation)
        return transform.apply(points)
