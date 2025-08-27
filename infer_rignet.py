import argparse
import os
from pprint import pprint

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
from torch.distributions import Normal
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm

# from datasets.shapenet_tris import TriangleNodesWithFaces
from datasets.rignet_2048 import TriangleNodesWithFaces
from diffusions.points_diffusion_dps_cd import GaussianDiffusion, get_betas

# from metrics.evaluation_metrics import EMD_CD, compute_all_metrics
# from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from models.dit3d_points_voxel import DiT3D_models
from models.pvcnn import PVCNN2Base
from utils.file_utils import *
from utils.training_utils import *

# from utils.misc import Evaluator
from utils.visualize import *

"""
models
"""


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


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - 0.5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(
        torch.max(1.0 - cdf_min, torch.ones_like(cdf_min) * 1e-12)
    )
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < 0.001,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12)),
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
    ]
    fp_blocks = [
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
        self,
        num_classes,
        embed_dim,
        use_att,
        dropout,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )


class MyModel(nn.Module):
    def __init__(
        self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str
    ):
        super(MyModel, self).__init__()
        self.diffusion = GaussianDiffusion(
            betas, loss_type, model_mean_type, model_var_type
        )

        self.pvcnn = PVCNN2(
            num_classes=512,  # we are finally needing this to be a feature vector
            embed_dim=args.embed_dim,
            use_att=args.attention,
            dropout=args.dropout,
            extra_feature_channels=0,
        )

        self.dit = DiT3D_models[args.model_type](
            pretrained=args.use_pretrained,
            num_points=args.npoints,
            in_channels=6,
            num_classes=args.num_classes,
            out_channels=51 if args.pe_output else 3,
        )

        self.pe = PE()

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, data_ref, cond, x0, y, clip_denoised=True):
        (
            total_bpd_b,
            vals_bt,
            prior_bpd_b,
            mse_bt,
            eps_mse_bt,
        ) = self.diffusion.calc_bpd_loop(
            self._denoise, data_ref, cond, x0, y, clip_denoised
        )

        return {
            "total_bpd_b": total_bpd_b,
            "terms_bpd": vals_bt,
            "prior_bpd_b": prior_bpd_b,
            "mse_bt": mse_bt,
            "eps_mse_bt": eps_mse_bt,
        }

    def _denoise(self, data, cond, t, y):
        B, D, N = data[:, :3].shape  # make sure the data is in Nx(3+3)
        data_in, data_ref = data[:, :3], data[:, -3:]
        data_cat = torch.cat([data_in, data_ref], dim=1)
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        ref_cond = self.pvcnn(data_cat, t)

        out = self.dit(data_cat, ref_cond, t, y)

        # assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, ref_pts, noises=None, y=None):
        B, D, N = data.shape
        # t = torch.randint(
        #     0, self.diffusion.num_timesteps, size=(B,), device=data.device
        # )
        num_small_steps = int(B * 0.5)
        num_big_steps = B - num_small_steps

        t_small = torch.randint(
            0,
            self.diffusion.num_timesteps // 5,  # first 200 steps for 50% of training
            size=(num_small_steps,),
            device=data.device,
        )

        t_big = torch.randint(
            self.diffusion.num_timesteps // 5,
            self.diffusion.num_timesteps,
            size=(num_big_steps,),
            device=data.device,
        )

        t = torch.cat([t_small, t_big], dim=0)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise,
            data_start=data,
            data_ref=ref_pts,
            cond=None,
            t=t,
            noise=noises,
            y=y,
        )
        # losses = losses * self.diffusion.get_noise_weights(t, t.device)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(
        self,
        shape,
        data_ref,
        cond,
        device,
        y,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,  # False
    ):
        return self.diffusion.p_sample_loop(
            self._denoise,
            shape=shape,
            data_ref=data_ref,
            cond=cond,
            device=device,
            y=y,
            noise_fn=noise_fn,
            clip_denoised=clip_denoised,
            keep_running=keep_running,
        )

    def gen_samples_dps(
        self,
        shape,
        data_ref,
        cond,
        device,
        y,
        ref_mesh_verts,
        ref_mesh_faces,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,  # False
    ):
        return self.diffusion.p_sample_loop_dps(
            self._denoise,
            shape=shape,
            data_ref=data_ref,
            cond=cond,
            device=device,
            y=y,
            ref_mesh_verts=ref_mesh_verts,
            ref_mesh_faces=ref_mesh_faces,
            noise_fn=noise_fn,
            clip_denoised=clip_denoised,
            keep_running=keep_running,
        )

    def gen_sample_traj(
        self,
        shape,
        data_ref,
        cond,
        device,
        y,
        freq,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
    ):
        return self.diffusion.p_sample_loop_trajectory(
            self._denoise,
            shape=shape,
            cond=cond,
            data_ref=data_ref,
            device=device,
            y=y,
            noise_fn=noise_fn,
            freq=freq,
            clip_denoised=clip_denoised,
            keep_running=keep_running,
        )  # [-1, 1]

    def gen_addnoise_traj(
        self,
        data_start,
        device,
        freq,
        noise_fn=torch.randn,
    ):
        return self.diffusion.q_sample_loop_trajectory(
            data_start,
            device,
            freq,
            noise_fn=noise_fn,
        )

    def train(self):
        self.dit.train()
        self.pvcnn.train()

    def eval(self):
        self.dit.eval()
        self.pvcnn.eval()

    def multi_gpu_wrapper(self, f):
        self.dit = f(self.dit)
        self.pvcnn = f(self.pvcnn)


def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    """

    :param target_shape_constraint: target voxels
    :return: constrained x
    """
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000) ** 2))

    def constrain_fn(x, t):
        eps_ = eps_all[t] if (t < 1000) else 0
        for _ in range(num_steps):
            x = x - eps_ * ((x - ground_truth) * mask)

        return x

    return constrain_fn


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


#############################################################################


def get_dataset(opt, npoints, category):
    tr_dataset = TriangleNodesWithFaces(
        config=opt,
        split="test",
        scale_augment=opt.scale_augment,
        shift_augment=opt.shift_augment,
        rot_augment=False,
        noise_augment=False,
        force_category=category,
        num_points=npoints,
    )
    te_dataset = TriangleNodesWithFaces(
        config=opt,
        split="test",  # "val"
        scale_augment=opt.scale_augment,
        shift_augment=opt.shift_augment,
        rot_augment=False,
        noise_augment=False,
        force_category=category,
        num_points=npoints,
    )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):
    if opt.distribution_type == "multi":
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=opt.world_size, rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=opt.world_size, rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = GeometricDataLoader(
        train_dataset,
        batch_size=opt.bs,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(opt.workers),
        drop_last=True,
    )

    if test_dataset is not None:
        test_dataloader = GeometricDataLoader(
            test_dataset,  # Corrected to use test_dataset instead of train_dataset
            batch_size=opt.bs,
            sampler=test_sampler,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=False,
        )
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def generate_eval(model, opt, gpu, outf_syn, evaluator):
    _, test_dataset = get_dataset(opt, opt.npoints, opt.category)

    _, test_dataloader, _, test_sampler = get_dataloader(
        opt, test_dataset, test_dataset
    )

    def new_y_chain(device, num_chain, num_classes):
        return torch.randint(low=0, high=num_classes, size=(num_chain,), device=device)

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    with torch.no_grad():  # no_grad
        samples = []

        for i, data in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            desc="Generating Samples",
        ):
            # if not (i in [0, 1, 4, 6, 10, 12, 13, 16, 18, 19, 27, 33, 34, 45, 79, 90, 98]):
            #     continue
            if i < opt.start_idx:
                continue
            ref_pts = data["face_pts"].transpose(1, 2) * 2.0  # [-1, 1]
            x = data["gt_pos"].transpose(1, 2) * 2.0
            # x = ref_pts.clone() # try if this is working
            y = data["cate_idx"]

            # mesh
            data_y = data["y"] / (opt.num_tokens - 3) - 0.5  # [-1, 1]

            for k in range(len(x)):
                coords = data_y[data["batch"] == k]
                coords = coords.view(-1, 9)
                pts = x.transpose(1, 2)[k] / 2.0

                gen_vertices, gen_faces = triangle_sequence_to_mesh(
                    coords.cpu().numpy().astype(np.float32)
                )
                plot_vertices_and_faces(
                    gen_vertices,
                    gen_faces,
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_mesh.jpg"),
                )
                plot_vertices_and_faces_and_points(
                    gen_vertices,
                    gen_faces,
                    pts.cpu().numpy(),
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_gt.jpg"),
                )
                plot_points(
                    pts.cpu().numpy(),
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_gt_pts.jpg"),
                )

            if opt.distribution_type == "multi" or (
                opt.distribution_type is None and gpu is not None
            ):
                x = x.cuda(gpu)
                y = y.cuda(gpu)
                ref_pts = ref_pts.cuda(gpu)
            elif opt.distribution_type == "single":
                x = x.cuda()
                y = y.cuda()
                ref_pts = ref_pts.cuda()

            # check all the kls
            # kl_dict = model.all_kl(ref_pts, None, x, y, clip_denoised=False)

            # eps_mse = kl_dict["eps_mse_bt"].mean(dim=0).cpu().numpy()
            # start_mse = kl_dict["mse_bt"].mean(dim=0).cpu().numpy()
            # # save each tensor in the dict into a npy file
            # # for k, v in kl_dict.items():
            # #     np.save(os.path.join(outf_syn, f"{i:03d}_{k}.npy"), v.cpu().numpy())

            # plot_error_curve_per_step(
            #     eps_mse,
            #     os.path.join(outf_syn, f"{i:03d}_{k:02d}_eps_perstep.jpg"),
            #     ylabel="eps_mse",
            #     )
            # plot_error_curve_per_step(
            #     start_mse,
            #     os.path.join(outf_syn, f"{i:03d}_{k:02d}_x0_perstep.jpg"),
            #     ylabel="start_mse",
            #     )

            """
            DEBUG
            """
            # t_max = 400
            # gen = (
            #     model.gen_samples_debug(
            #         x,
            #         ref_pts,
            #         None,
            #         x.device,
            #         y,
            #         t_max,
            #         clip_denoised=False,
            #         keep_running=True,
            #     )
            #     .detach()
            #     .cpu()
            # )

            gen = (
                model.gen_samples_dps(
                    new_x_chain(x, len(x)).shape,
                    ref_pts,
                    None,
                    x.device,
                    y,
                    ref_mesh_verts=gen_vertices * 2.0,
                    ref_mesh_faces=gen_faces,
                    clip_denoised=False,
                    keep_running=False,
                )
                .detach()
                .cpu()
            )

            # gen = (
            #     model.gen_samples(
            #         new_x_chain(x, len(x)).shape,
            #         ref_pts,
            #         None,
            #         x.device,
            #         y,
            #         clip_denoised=False,
            #         keep_running=False,
            #     )
            #     .detach()
            #     .cpu()
            # )

            gen = gen.transpose(1, 2).contiguous()
            x = x.transpose(1, 2).contiguous()

            samples.append(gen.to(gpu).contiguous())

            for k, gen_pts in enumerate(gen):
                coords = data_y[data["batch"] == k]
                coords = coords.view(-1, 9)

                gen_vertices, gen_faces = triangle_sequence_to_mesh(
                    coords.cpu().numpy().astype(np.float32)
                )
                error_map = np.linalg.norm(
                    gen_pts.cpu().numpy() - x[k].cpu().numpy(), axis=1
                )

                plot_ref_gt = ref_pts.transpose(1, 2)[k] / 2.0

                plot_gen_pts = gen_pts / 2.0

                # rotate the plot_ref_gt around x axis for 90 degrees

                plot_error_map(
                    error_map,
                    plot_ref_gt.cpu().numpy(),
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_error.jpg"),
                )
                plot_vertices_and_faces_and_points(
                    gen_vertices,
                    gen_faces,
                    plot_gen_pts.cpu().numpy(),
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_sample.jpg"),
                )
                plot_points(
                    plot_gen_pts.cpu().numpy(),
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_gen_pts.jpg"),
                )
                # save the mesh into obj
                trimesh.Trimesh(gen_vertices, gen_faces).export(
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_mesh.obj")
                )
                # save the generated points as ply
                write_np_to_ply(
                    plot_gen_pts.cpu().numpy(),
                    os.path.join(outf_syn, f"{i:03d}_{k:02d}_genpts.ply"),
                )

            # visualize_pointcloud_batch(
            #     os.path.join(outf_syn, f"{i}.png"), gen, None, None, None
            # )

            # Compute metrics
            # results = compute_all_metrics(gen, x, opt.bs)
            # results = {
            #     k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            #     for k, v in results.items()
            # }

            # jsd = JSD(gen.numpy(), x.numpy())

            # evaluator.update(results, jsd)

        # stats = evaluator.finalize_stats()
        stats = 1

        # samples = torch.cat(samples, dim=0)
        # samples_gather = concat_all_gather(samples)

        # torch.save(samples_gather, opt.eval_path)

    return stats


def main(opt):
    opt.beta_start = 3e-7  # 1e-7
    opt.beta_end = 0.1  # 0.1
    opt.schedule_type = "exp"  # "warm0.5"
    opt.model_mean_type = "xstart"

    output_dir = get_output_dir(opt.model_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    opt.dist_url = f"tcp://{opt.node}:{opt.port}"
    print("Using url {}".format(opt.dist_url))

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        test(opt.gpu, opt, output_dir)


def test(gpu, opt, output_dir):
    logger = setup_logging(output_dir)

    if opt.distribution_type == "multi":
        should_diag = gpu == 0
    else:
        should_diag = True

    (outf_syn,) = setup_output_subdirs(output_dir, "syn_firstx")  # "syn"

    if opt.distribution_type == "multi":
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank = opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank,
        )

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

    """
    create networks
    """

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = MyModel(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == "multi":  # Multiple processes, single GPU per process

        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu
            )

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)

    elif opt.distribution_type == "single":

        def _transform_(m):
            return nn.parallel.DataParallel(m)

        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError("distribution_type = multi | single | None")

    if should_diag:
        logger.info(opt)

        logger.info("Model = %s" % str(model))
        total_params = sum(param.numel() for param in model.parameters()) / 1e6
        logger.info("Total_params = %s MB " % str(total_params))  # S4: 32.81 MB

    model.eval()

    # evaluator = Evaluator(results_dir=output_dir)
    evaluator = None

    with torch.no_grad():
        if should_diag:
            logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param["model_state"], strict=False)
        # model.train()

        opt.eval_path = os.path.join(outf_syn, "samples.pth")
        Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)

        stats = generate_eval(model, opt, gpu, outf_syn, evaluator)

        if should_diag:
            logger.info(stats)


if __name__ == "__main__":
    opt = parse_args()
    set_seed(opt)

    main(opt)
