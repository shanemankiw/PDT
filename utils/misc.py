import os

import numpy as np


class Evaluator(object):
    def __init__(self, results_dir="./results"):
        super(Evaluator, self).__init__()

        self.cd1nn_list = []
        self.emd1nn_list = []
        self.cdcov_list = []
        self.emdcov_list = []
        self.cdmmd_list = []
        self.emdmmd_list = []

        self.jsd_list = []

        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir

    def finalize_stats(self):
        (
            cd1nn_full_list,
            emd1nn_full_list,
            cdcov_full_list,
            emdcov_full_list,
            cdmmd_full_list,
            emdmmd_full_list,
            jsd_full_list,
        ) = self.gather_results()

        stats = {}

        stats["1-NNA-CD"] = np.mean(cd1nn_full_list)
        stats["1-NNA-EMD"] = np.mean(emd1nn_full_list)
        stats["COV-CD"] = np.mean(cdcov_full_list)
        stats["COV-EMD"] = np.mean(emdcov_full_list)
        stats["MMD-CD"] = np.mean(cdmmd_full_list)
        stats["MMD-EMD"] = np.mean(emdmmd_full_list)

        stats["JSD"] = np.mean(jsd_full_list)

        return stats

    def gather_results(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            return (
                self.cd1nn_list,
                self.emd1nn_list,
                self.cdcov_list,
                self.emdcov_list,
                self.cdmmd_list,
                self.emdmmd_list,
                self.jsd_list,
            )
        world_size = dist.get_world_size()

        cd1nn_list = [None for _ in range(world_size)]
        dist.all_gather_object(cd1nn_list, self.cd1nn_list)
        cd1nn_list = [x for cd1nn in cd1nn_list for x in cd1nn]

        emd1nn_list = [None for _ in range(world_size)]
        dist.all_gather_object(emd1nn_list, self.emd1nn_list)
        emd1nn_list = [x for emd1nn in emd1nn_list for x in emd1nn]

        cdcov_list = [None for _ in range(world_size)]
        dist.all_gather_object(cdcov_list, self.cdcov_list)
        cdcov_list = [x for cdcov in cdcov_list for x in cdcov]

        emdcov_list = [None for _ in range(world_size)]
        dist.all_gather_object(emdcov_list, self.emdcov_list)
        emdcov_list = [x for emdcov in emdcov_list for x in emdcov]

        cdmmd_list = [None for _ in range(world_size)]
        dist.all_gather_object(cdmmd_list, self.cdmmd_list)
        cdmmd_list = [x for cdmmd in cdmmd_list for x in cdmmd]

        emdmmd_list = [None for _ in range(world_size)]
        dist.all_gather_object(emdmmd_list, self.emdmmd_list)
        emdmmd_list = [x for emdmmd in emdmmd_list for x in emdmmd]

        jsd_list = [None for _ in range(world_size)]
        dist.all_gather_object(jsd_list, self.jsd_list)
        jsd_list = [x for jsd in jsd_list for x in jsd]

        return (
            cd1nn_list,
            emd1nn_list,
            cdcov_list,
            emdcov_list,
            cdmmd_list,
            emdmmd_list,
            jsd_list,
        )

    def clear(self):
        self.cd1nn_list = []
        self.emd1nn_list = []
        self.cdcov_list = []
        self.emdcov_list = []
        self.cdmmd_list = []
        self.emdmmd_list = []
        self.jsd_list = []

    def update(self, results, jsd):
        cd1nn = results["1-NN-CD-acc"]
        emd1nn = results["1-NN-EMD-acc"]

        cdcov = results["lgan_cov-CD"]
        emdcov = results["lgan_cov-EMD"]

        cdmmd = results["lgan_mmd-CD"]
        emdmmd = results["lgan_mmd-EMD"]

        # Save
        self.cd1nn_list.append(cd1nn)
        self.emd1nn_list.append(emd1nn)
        self.cdcov_list.append(cdcov)
        self.emdcov_list.append(emdcov)
        self.cdmmd_list.append(cdmmd)
        self.emdmmd_list.append(emdmmd)
        self.jsd_list.append(jsd)


def scale_vertices(
    vertices,
    x_lims=(0.75, 1.25),
    y_lims=(0.75, 1.25),
    z_lims=(0.75, 1.25),
):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack(
        [vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1
    )
    return vertices


def scale_vertices_w_points(
    vertices,
    points,
    x_lims=(0.75, 1.25),
    y_lims=(0.75, 1.25),
    z_lims=(0.75, 1.25),
):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack(
        [vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1
    )
    points = np.stack([points[:, 0] * x, points[:, 1] * y, points[:, 2] * z], axis=-1)
    return vertices, points


def scale_vertices_w_inputs(
    vertices,
    inputs,
    x_lims=(0.75, 1.25),
    y_lims=(0.75, 1.25),
    z_lims=(0.75, 1.25),
):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack(
        [vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1
    )
    points = np.stack([points[:, 0] * x, points[:, 1] * y, points[:, 2] * z], axis=-1)
    return vertices, points


def shift_vertices(
    vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)
):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack(
        [vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1
    )
    return vertices


def shift_points_w_offsets(vertices, x, y, z):
    # shift x, y, z

    vertices = np.stack(
        [vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1
    )
    return vertices


def shift_vertices_w_points(
    vertices, points, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)
):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack(
        [vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1
    )
    points = np.stack([points[:, 0] + x, points[:, 1] + y, points[:, 2] + z], axis=-1)
    return vertices, points


def jitter_vertices(
    vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)
):
    x_shifts = np.random.uniform(
        low=x_lims[0], high=x_lims[1], size=(vertices.shape[0],)
    )
    y_shifts = np.random.uniform(
        low=y_lims[0], high=y_lims[1], size=(vertices.shape[0],)
    )
    z_shifts = np.random.uniform(
        low=z_lims[0], high=z_lims[1], size=(vertices.shape[0],)
    )

    shifts = np.stack([x_shifts, y_shifts, z_shifts], axis=-1)

    return vertices + shifts


def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices


def normalize_points_w_bounds(points, bounds):
    points = points - (bounds[0] + bounds[1])[None, :] / 2
    points = points / (bounds[1] - bounds[0]).max()
    return points


def normalize_vertices_w_points(vertices, points):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    points = points - (bounds[0] + bounds[1])[None, :] / 2
    points = points / (bounds[1] - bounds[0]).max()
    return vertices, points
