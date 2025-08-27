import sys

sys.path.append("/home/jhwang/dit-points")
import json
import os
import pickle
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import omegaconf
import torch
import torch.utils.data
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, default_collate
from torch_cluster import fps
from torch_geometric.data import Batch
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader.dataloader import Collater as GeometricCollator
from tqdm import tqdm

from datasets import (
    fill_and_perm,
    quantize_coordinates,
    sort_vertices_and_faces,
    sort_vertices_and_faces_wpoints,
)
from utils.misc import (
    normalize_points_w_bounds,
)
from utils.rignet import parse_joints_info
from utils.visualize import *


# debug
def write_np_to_ply(verts, path):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for vert in verts:
            f.write(f"{vert[0]} {vert[1]} {vert[2]}\n")


class TriangleNodes(GeometricDataset):
    def __init__(
        self,
        config,
        split,
        scale_augment,
        shift_augment,
        rot_augment,
        noise_augment,
        force_category,
        num_points,
        use_start_stop=False,
        only_backward_edges=False,
    ):
        super().__init__()
        data_path = Path(config.dataroot)
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.cached_gt_verts = []
        self.cached_dense_pcs = []
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.rot_augment = rot_augment
        self.noise_augment = noise_augment
        self.low_augment = config.low_augment
        self.use_start_stop = use_start_stop
        self.ce_output = config.ce_output
        self.only_backward_edges = only_backward_edges
        self.num_tokens = config.num_tokens - 3
        self.pad_to_faces = config.pad_to_faces
        self.num_points = num_points
        self.mask_pct = [0.0, 0.0]  # [0.6, 0.8] [0.2, 0.4] [0.0, 0.0]

        pkl_file = f"data/RigNet/{split}_final_rot90.pkl"
        
        """
        Comment this part if pkl already exists
        """
        # list_txt = f"data/RigNet/{split}_final.txt"
        
        # all_names = []
        # all_joints = []
        # all_dense_pc = []
        # all_mesh_faces = []
        # all_mesh_vertices = []

        # with open(list_txt, "r") as f:
        #     lines = f.readlines()
        
        # for line in tqdm(lines):
        #     name = line.strip()
        #     all_names.append(name)

        #     joints = parse_joints_info(f"data/RigNet/rig_info_remesh/{name}.txt")
            
        #     mesh = trimesh.load(f"data/RigNet/obj_remesh/{name}.obj")
        #     # rotate mesh 90 degrees around x axis
        #     rot_mat = np.array(
        #         [
        #             [1, 0, 0],
        #             [0, 0, -1],
        #             [0, 1, 0],
        #         ]
        #     )
        #     vertices = mesh.vertices
        #     vertices = vertices @ rot_mat.T
        #     joints = joints @ rot_mat.T

        #     # center
        #     center = vertices.mean(axis=0)
        #     vertices -= center
        #     joints -= center

        #     # normalize
        #     bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        #     vertices = normalize_points_w_bounds(vertices, bounds) * 2.0
        #     mesh.vertices = vertices
        #     joints = normalize_points_w_bounds(joints, bounds) * 2.0

        #     # mesh.export(f"data/RigNet/obj_remesh_rot/{name}_rot90.obj")
        #     # write_np_to_ply(joints, f"data/RigNet/ply_remesh_rot/{name}_rot90.ply")

        #     # sample 20k points evenly on the mesh
        #     points = sample_facepts_from_mesh_even(
        #         mesh.vertices, mesh.faces, num_sample=20000
        #     )
        #     all_dense_pc.append(points)
        #     all_joints.append(joints)

        #     all_mesh_faces.append(mesh.faces)
        #     all_mesh_vertices.append(mesh.vertices)
        
        # with open(pkl_file, "wb") as f:
        #     pickle.dump(
        #         {
        #             "names": all_names,
        #             "joints": all_joints,
        #             "dense_pc": all_dense_pc,
        #             "mesh_faces": all_mesh_faces,
        #             "mesh_vertices": all_mesh_vertices,
        #         },
        #         f,
        #     )
        
        # load the pkl file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        
        self.names = data["names"]
        self.cached_gt_verts = data["joints"]
        self.cached_dense_pcs = data["dense_pc"]
        self.cached_faces = data["mesh_faces"]
        self.cached_vertices = data["mesh_vertices"]

        print("Loaded {} data from pkl".format(len(self.cached_dense_pcs)))

    def len(self):
        return len(self.cached_vertices)

    def get_all_features_for_shape(self, idx):
        vertices = np.array(self.cached_vertices[idx])
        good_verts = np.array(self.cached_gt_verts[idx])
        dense_pc = np.array(self.cached_dense_pcs[idx])
        faces = np.array(self.cached_faces[idx])
        # cat = np.zeros_like(faces[:, 0]).astype(int)
        cat = torch.tensor([0]).long()

        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            sx = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
            sy = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
            sz = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
            vertices = np.stack(
                [vertices[:, 0] * sx, vertices[:, 1] * sy, vertices[:, 2] * sz], axis=-1
            )
            dense_pc = np.stack(
                [dense_pc[:, 0] * sx, dense_pc[:, 1] * sy, dense_pc[:, 2] * sz], axis=-1
            )
            good_verts = np.stack(
                [good_verts[:, 0] * sx, good_verts[:, 1] * sy, good_verts[:, 2] * sz],
                axis=-1,
            )
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = normalize_points_w_bounds(vertices, bounds)
        dense_pc = normalize_points_w_bounds(dense_pc, bounds)
        good_verts = normalize_points_w_bounds(good_verts, bounds)

        if self.shift_augment:
            x_lims, y_lims, z_lims = (-0.1, 0.1), (-0.1, 0.1), (-0.075, 0.075)
            x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
            y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
            z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
            x = max(min(x, 0.5 - good_verts[:, 0].max()), -0.5 - good_verts[:, 0].min())
            y = max(min(y, 0.5 - good_verts[:, 1].max()), -0.5 - good_verts[:, 1].min())
            z = max(min(z, 0.5 - good_verts[:, 2].max()), -0.5 - good_verts[:, 2].min())
            vertices = np.stack(
                [vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1
            )
            dense_pc = np.stack(
                [dense_pc[:, 0] + x, dense_pc[:, 1] + y, dense_pc[:, 2] + z], axis=-1
            )
            good_verts = np.stack(
                [good_verts[:, 0] + x, good_verts[:, 1] + y, good_verts[:, 2] + z],
                axis=-1,
            )
        if self.rot_augment:
            # rotate around axis z
            rnd_angle = np.random.uniform(low=0, high=2 * np.pi)
            rot_mat = np.array(
                [
                    [np.cos(rnd_angle), -np.sin(rnd_angle), 0],
                    [np.sin(rnd_angle), np.cos(rnd_angle), 0],
                    [0, 0, 1],
                ]
            )
            vertices = vertices @ rot_mat.T
            dense_pc = dense_pc @ rot_mat.T
            good_verts = good_verts @ rot_mat.T

        # quantize the verts
        # vertices = np.clip((vertices + 0.5), 0, 1) * self.num_tokens
        # vertices_quantized = vertices.round().astype(int)
        # vertices = vertices_quantized / self.num_tokens - 0.5

        # good_verts = np.clip((good_verts + 0.5), 0, 1) * self.num_tokens
        # good_verts_quantized = good_verts.round().astype(int)
        # good_verts = good_verts_quantized / self.num_tokens - 0.5

        triangles = vertices[faces].reshape(-1, 9)
        # other_feats = np.random.rand(triangles.shape[0], 3 + 1 + 3)

        features = triangles

        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # sample points
        # face_pts = farthest_point_sampling(dense_pc, self.num_points, return_idx=False)
        if dense_pc.shape[0] < self.num_points:
            diff_pts = np.random.choice(
                dense_pc.shape[0], self.num_points - dense_pc.shape[0], replace=True
            )
            face_pts = np.concatenate([dense_pc, dense_pc[diff_pts]], axis=0)
        else:
            rnd_idx = np.random.choice(
                dense_pc.shape[0], self.num_points, replace=False
            )
            face_pts = dense_pc[rnd_idx]

        # sort points by z-y-x
        z_coords = face_pts[:, 2]
        y_coords = face_pts[:, 1]
        x_coords = face_pts[:, 0]

        # Use numpy.lexsort to sort by z, then y, then x
        sorted_indices = np.lexsort((x_coords, y_coords, z_coords))
        face_pts = face_pts[sorted_indices][None, :]

        if self.noise_augment:
            noise = np.random.normal(loc=0, scale=0.001, size=face_pts.shape)
            face_pts += noise

        # find for each face pt, its closest vertice
        gt_pos, gt_idx = get_gt_pos_closest(face_pts[0], good_verts)
        gt_pos = gt_pos[None, :]

        # gt_diff = gt_pos - face_pts

        face_neighborhood = np.array(mesh_trimesh.face_neighborhood)  # type: ignore

        # DEBUG
        # # save the mesh
        # mesh_trimesh.export(f"viz/mesh_{idx}.obj")
        # # save dense pc, and joints
        # write_np_to_ply(dense_pc, f"viz/dense_pc_{idx}.ply")
        # write_np_to_ply(good_verts, f"viz/joints_{idx}.ply")
        # write_np_to_ply(gt_pos[0], f"viz/gt_pos_{idx}.ply")

        target = torch.from_numpy(features[:, :9]).float()  # target is in-order

        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        return (
            features,
            target,
            vertices,
            faces,
            face_neighborhood,
            face_pts,
            gt_pos,
            cat,
        )

    def get(self, idx):
        (
            features,
            target,
            _,
            _,
            face_neighborhood,
            face_pts,
            gt_pos,
            cat,
        ) = self.get_all_features_for_shape(idx)
        return GeometricData(
            x=torch.from_numpy(features).float(),
            y=target,
            edge_index=torch.from_numpy(face_neighborhood.T).long(),
            face_pts=torch.from_numpy(face_pts).float(),
            gt_pos=torch.from_numpy(gt_pos).float(),
            cate_idx=cat,
            idx=idx,
        )


class TriangleNodesWithFaces(TriangleNodes):
    def __init__(
        self,
        config,
        split,
        scale_augment,
        shift_augment,
        rot_augment,
        noise_augment,
        force_category,
        num_points,
    ):
        super().__init__(
            config,
            split,
            scale_augment,
            shift_augment,
            rot_augment,
            noise_augment,
            force_category,
            num_points,
        )

    def get(self, idx):
        (
            features,
            target,
            vertices,
            faces,
            face_neighborhood,
            face_pts,
            gt_pos,
            cat,
        ) = self.get_all_features_for_shape(idx)
        return GeometricData(
            x=torch.from_numpy(features).float(),
            y=target,
            edge_index=torch.from_numpy(face_neighborhood.T).long(),
            num_vertices=vertices.shape[0],
            faces=torch.from_numpy(np.array(faces)).long(),
            face_pts=torch.from_numpy(face_pts).float(),
            gt_pos=torch.from_numpy(gt_pos).float(),
            cate_idx=cat,
            idx=idx,
        )


class TriangleNodesWithFacesDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        follow_batch=None,
        exclude_keys=None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)
        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=FaceCollator(follow_batch, exclude_keys),
            **kwargs,
        )


class FaceCollator(GeometricCollator):
    def __init__(self, follow_batch, exclude_keys):
        super().__init__(follow_batch, exclude_keys)

    def __call__(self, batch):
        elem = batch[0]

        num_vertices = 0
        for b_idx in range(len(batch)):
            mask = batch[b_idx].fill_mask
            batch[b_idx].faces[mask] += num_vertices
            num_vertices += batch[b_idx].num_vertices

        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate(self, batch):  # pragma: no cover
        raise NotImplementedError


def get_gt_pos(face_pts, vertices):
    # Step 1: Initial Nearest Neighbor Search
    kdtree = cKDTree(vertices)
    distances, indices = kdtree.query(face_pts)
    initial_gt_pts = vertices[indices]

    # Construct the cost matrix (distance matrix)
    cost_matrix = cdist(face_pts, vertices, "euclidean")

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the final gt_pts array
    initial_gt_pts[row_ind] = vertices[col_ind]

    return initial_gt_pts, col_ind


def get_gt_pos_closest(face_pts, vertices):
    # Build a KD-tree from the verts array
    kdtree = cKDTree(vertices)

    # Query the KD-tree to find the nearest neighbor for each point in face_pts
    distances, indices = kdtree.query(face_pts)
    gt_pts = vertices[indices]

    return gt_pts, indices


def farthest_point_sampling(points, num_samples, return_idx=False):
    """
    Perform farthest point sampling on the point cloud
    :param points: numpy array of shape (N, 3)
    :param num_samples: number of points to sample
    :return: numpy array of sampled points
    """
    points_tensor = torch.tensor(
        points, dtype=torch.float
    )  # Convert to tensor and add batch dimension
    sampled_indices = fps(
        points_tensor, ratio=num_samples / points.shape[0], random_start=True
    )
    sampled_points = points[sampled_indices.numpy()]
    if return_idx:
        return sampled_points, sampled_indices.numpy()
    return sampled_points


def sample_facepts_from_mesh(vertices, tri_faces, num_sample=2048):
    """
    given a mesh, sample num_sample points on the surface
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=tri_faces)

    surface_points, _ = trimesh.sample.sample_surface(
        mesh,
        num_sample,
    )

    return surface_points


def sample_facepts_from_mesh_even(vertices, tri_faces, num_sample=2048):
    """
    given a mesh, sample num_sample points on the surface
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=tri_faces)

    surface_points, _ = trimesh.sample.sample_surface_even(
        mesh,
        num_sample,
    )

    return surface_points


def normal(triangles):
    # The cross product of two sides is a normal vector
    if torch.is_tensor(triangles):
        return torch.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1
        )
    else:
        return np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1
        )


def area(triangles):
    # The norm of the cross product of two sides is twice the area
    if torch.is_tensor(triangles):
        return torch.norm(normal(triangles), dim=1) / 2
    else:
        return np.linalg.norm(normal(triangles), axis=1) / 2


def angle(triangles):
    v_01 = triangles[:, 1] - triangles[:, 0]
    v_02 = triangles[:, 2] - triangles[:, 0]
    v_10 = -v_01
    v_12 = triangles[:, 2] - triangles[:, 1]
    v_20 = -v_02
    v_21 = -v_12
    if torch.is_tensor(triangles):
        return torch.stack(
            [
                angle_between(v_01, v_02),
                angle_between(v_10, v_12),
                angle_between(v_20, v_21),
            ],
            dim=1,
        )
    else:
        return np.stack(
            [
                angle_between(v_01, v_02),
                angle_between(v_10, v_12),
                angle_between(v_20, v_21),
            ],
            axis=1,
        )


def angle_between(v0, v1):
    v0_u = unit_vector(v0)
    v1_u = unit_vector(v1)
    if torch.is_tensor(v0):
        return torch.arccos(torch.clip(torch.einsum("ij,ij->i", v0_u, v1_u), -1.0, 1.0))
    else:
        return np.arccos(np.clip(np.einsum("ij,ij->i", v0_u, v1_u), -1.0, 1.0))


def unit_vector(vector):
    if torch.is_tensor(vector):
        return vector / (torch.norm(vector, dim=-1)[:, None] + 1e-8)
    else:
        return vector / (np.linalg.norm(vector, axis=-1)[:, None] + 1e-8)


def create_feature_stack(vertices, faces, num_tokens):
    vertices, faces = sort_vertices_and_faces(vertices, faces, num_tokens)
    vertices = vertices[:, [2, 1, 0]]  # work around
    vertices[:, 0] = -vertices[:, 0]
    # save the vertices
    # debug
    # write_np_to_ply(vertices, "viz/vertices_c.ply")
    # plot_points(vertices, "viz/vertices_c.png")
    # plot_vertices_and_faces_and_points(vertices, faces, vertices, 'viz/mesh_c.png')
    triangles = vertices[faces, :]
    triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
    return triangles, normals, areas, angles, vertices, faces


def create_feature_stack_wpoints(vertices, faces, face_pts, num_tokens):
    vertices, faces, face_pts = sort_vertices_and_faces_wpoints(
        vertices, faces, face_pts, num_tokens
    )
    # need more features: positions, angles, area, cross_product
    triangles = vertices[faces, :]
    triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
    return triangles, normals, areas, angles, vertices, faces, face_pts


def create_feature_stack_from_triangles(triangles):
    t_areas = area(triangles) * 1e3
    t_angles = angle(triangles) / float(np.pi)
    t_normals = unit_vector(normal(triangles))
    return (
        triangles.reshape(-1, 9),
        t_normals.reshape(-1, 3),
        t_areas.reshape(-1, 1),
        t_angles.reshape(-1, 3),
    )


def sample_pts_from_edges(edge_coordinates, edge_directions, edge_lengths, N):
    total_length = np.sum(edge_lengths)

    # Calculate the number of points to sample from each edge
    points_distribution = (edge_lengths / total_length) * N
    points_per_edge = np.floor(points_distribution).astype(int)

    # Distribute the remaining points (due to floor operation)
    remaining_points = N - np.sum(points_per_edge)
    if remaining_points > 0:
        indices = np.argsort(points_distribution - points_per_edge)[
            : int(remaining_points)
        ]
        points_per_edge[indices] += 1

    # Calculate the cumulative sum of points to determine the indices
    cumulative_points = np.zeros_like(points_per_edge)
    np.cumsum(points_per_edge, out=cumulative_points)

    # Generate the t values for interpolation
    t_values = np.linspace(0, 1, N, endpoint=False)

    # Determine which edge each point belongs to
    edge_indices = np.searchsorted(cumulative_points, np.arange(N), side="right")

    # Calculate the start points and directions for each sampled point
    start_points = edge_coordinates[:, 0][edge_indices]
    sampled_directions = edge_directions[edge_indices]

    # Calculate the distances each point needs to be from the start point along the edge
    distances = (t_values * edge_lengths[edge_indices]) / points_per_edge[edge_indices]

    # Sample points
    sampled_points = start_points + sampled_directions * distances[:, np.newaxis]

    return sampled_points, sampled_directions


def get_edge_attributes(mesh, N=512):
    """
    mesh should be a trimesh
    """
    edges = mesh.edges_unique

    edge_coordinates = mesh.vertices[edges]

    edge_centers = np.mean(edge_coordinates, axis=1)

    edge_directions = np.diff(edge_coordinates, axis=1)[:, 0]

    edge_lengths = np.linalg.norm(edge_directions, axis=1)

    # Normalize the edge directions if needed
    edge_directions = edge_directions / edge_lengths[:, None]

    edge_points_sample, edge_dirs_sample = sample_pts_from_edges(
        edge_coordinates, edge_directions, edge_lengths, N
    )

    return (
        edge_centers,
        edge_directions,
        edge_lengths,
        edge_points_sample,
        edge_dirs_sample,
    )


if __name__ == "__main__":
    # test out the dataset
    from utils.training_utils import *

    opt = parse_args()
    opt.beta_start = 3e-7  # 1e-7
    opt.beta_end = 0.1  # 0.3
    opt.schedule_type = "exp"  # "warm0.5"
    opt.model_mean_type = "eps"
    dataset = TriangleNodesWithFaces(
        config=opt,
        split="train",
        scale_augment=False,
        shift_augment=False,
        force_category=None,
        num_points=2048,
    )
    test_loader = TriangleNodesWithFacesDataloader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    for i, data in enumerate(test_loader):
        a = 1
