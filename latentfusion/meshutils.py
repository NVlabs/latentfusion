import typing

import numpy as np
from scipy import linalg

import trimesh
import trimesh.remesh
from trimesh.visual.material import SimpleMaterial

EPS = 10e-10


def compute_vertex_normals(vertices, faces):
    normals = np.ones_like(vertices)
    triangles = vertices[faces]
    triangle_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                triangles[:, 2] - triangles[:, 0])
    triangle_normals /= (linalg.norm(triangle_normals, axis=1)[:, None] + EPS)
    normals[faces[:, 0]] += triangle_normals
    normals[faces[:, 1]] += triangle_normals
    normals[faces[:, 2]] += triangle_normals
    normals /= (linalg.norm(normals, axis=1)[:, None] + 0)

    return normals


def are_trimesh_normals_corrupt(trimesh):
    corrupt_normals = linalg.norm(trimesh.vertex_normals, axis=1) == 0.0
    return corrupt_normals.sum() > 0


def subdivide_mesh(mesh):
    attributes = {}
    if hasattr(mesh.visual, 'uv'):
        attributes = {'uv': mesh.visual.uv}
    vertices, faces, attributes = trimesh.remesh.subdivide(
        mesh.vertices, mesh.faces, attributes=attributes)
    mesh.vertices = vertices
    mesh.faces = faces
    if 'uv' in attributes:
        mesh.visual.uv = attributes['uv']

    return mesh


class Object3D(object):
    """Represents a graspable object."""

    def __init__(self, path, load_materials=False):
        scene = trimesh.load(str(path))
        if isinstance(scene, trimesh.Trimesh):
            scene = trimesh.Scene(scene)

        self.meshes: typing.List[trimesh.Trimesh] = list(scene.dump())

        self.path = path
        self.scale = 1.0

    def to_scene(self):
        return trimesh.Scene(self.meshes)

    def are_normals_corrupt(self):
        for mesh in self.meshes:
            if are_trimesh_normals_corrupt(mesh):
                return True

        return False

    def recompute_normals(self):
        for mesh in self.meshes:
            mesh.vertex_normals = compute_vertex_normals(mesh.vertices, mesh.faces)

        return self

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
        self.scale = scale
        for mesh in self.meshes:
            mesh.apply_scale(self.scale)

        return self

    def resize(self, size, ref='diameter'):
        """Set longest of all three lengths in Cartesian space.

        :param size
        """
        if ref == 'diameter':
            ref_scale = self.bounding_diameter
        else:
            ref_scale = self.bounding_size

        self.scale = size / ref_scale
        for mesh in self.meshes:
            mesh.apply_scale(self.scale)

        return self

    @property
    def centroid(self):
        return self.bounds.mean(axis=0)

    @property
    def bounding_size(self):
        return max(self.extents)

    @property
    def bounding_diameter(self):
        centroid = self.bounds.mean(axis=0)
        max_radius = linalg.norm(self.vertices - centroid, axis=1).max()
        return max_radius * 2

    @property
    def bounding_radius(self):
        return self.bounding_diameter / 2.0

    @property
    def extents(self):
        min_dim = np.min(self.vertices, axis=0)
        max_dim = np.max(self.vertices, axis=0)
        return max_dim - min_dim

    @property
    def bounds(self):
        min_dim = np.min(self.vertices, axis=0)
        max_dim = np.max(self.vertices, axis=0)
        return np.stack((min_dim, max_dim), axis=0)

    def recenter(self, method='bounds'):
        if method == 'mean':
            # Center the mesh.
            vertex_mean = np.mean(self.vertices, 0)
            translation = -vertex_mean
        elif method == 'bounds':
            center = self.bounds.mean(axis=0)
            translation = -center
        else:
            raise ValueError(f"Unknown method {method!r}")

        for mesh in self.meshes:
            mesh.apply_translation(translation)

        return self

    @property
    def vertices(self):
        return np.concatenate([mesh.vertices for mesh in self.meshes])
