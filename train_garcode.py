from copy import deepcopy

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# from tensorboardX import SummaryWriter
from torch.distributions import Normal
from torch_geometric.data import DataLoader as GeometricDataLoader

import wandb
from datasets.garcode_2048 import TriangleNodesWithFaces
from diffusions.points_diffusion import GaussianDiffusion, get_betas
from models.dit3d_points_voxel import DiT3D_models
from models.pvcnn import PVCNN2Base

# from models.dit3d_points_voxel import DiT3D_voxel_models
from utils.file_utils import *
from utils.training_utils import *
from utils.visualize import *

"""
models
"""

class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
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

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
            self._denoise, x0, clip_denoised
        )

        return {
            "total_bpd_b": total_bpd_b,
            "terms_bpd": vals_bt,
            "prior_bpd_b": prior_bpd_b,
            "mse_bt": mse_bt,
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
        t = torch.randint(
            0, self.diffusion.num_timesteps, size=(B,), device=data.device
        )
        # num_small_steps = int(B * 0.5)
        # num_big_steps = B - num_small_steps

        # t_small = torch.randint(
        #     0,
        #     self.diffusion.num_timesteps // 5,  # first 200 steps for 50% of training
        #     size=(num_small_steps,),
        #     device=data.device,
        # )

        # t_big = torch.randint(
        #     self.diffusion.num_timesteps // 5,
        #     self.diffusion.num_timesteps,
        #     size=(num_big_steps,),
        #     device=data.device,
        # )

        # t = torch.cat([t_small, t_big], dim=0)

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
            aux_weight=1e-4,
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
        keep_running=False,
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


def get_dataset(opt, npoints, category):
    tr_dataset = TriangleNodesWithFaces(
        config=opt,
        split="train",  # "train"
        scale_augment=opt.scale_augment,
        shift_augment=opt.shift_augment,
        rot_augment=opt.rot_augment,
        noise_augment=opt.noise_augment,
        force_category=category,
        num_points=npoints,
    )
    te_dataset = TriangleNodesWithFaces(
        config=opt,
        split="val",  # "val"
        scale_augment=False,
        shift_augment=False,
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
            batch_size=opt.bs,  # test data just bs=2
            sampler=test_sampler,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=False,
        )
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def get_random_batch(data_loader):
    data_iter = iter(data_loader)
    return next(data_iter)


def train(gpu, opt, output_dir, noises_init):
    set_seed(opt)
    logger = setup_logging(output_dir)

    if not opt.debug:
        if opt.use_tb:
            # tb writers
            tb_writer = SummaryWriter(output_dir)
        if opt.use_wandb and gpu == 1:
            wandb.init(project="dit-points", name=opt.experiment_name)

    if opt.distribution_type == "multi":
        should_diag = gpu == 0
    else:
        should_diag = True
    if should_diag or gpu == 1:
        (outf_syn,) = setup_output_subdirs(output_dir, "syn")

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

        opt.saveIter = int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)
        opt.evalIter = int(opt.evalIter / opt.ngpus_per_node)

    """ data """
    train_dataset, val_dataset = get_dataset(opt, opt.npoints, opt.category)
    dataloader, test_dataloader, train_sampler, test_sampler = get_dataloader(
        opt, train_dataset, val_dataset
    )

    """
    create networks
    """

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = MyModel(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    # Note that parameter initialization is done within the DiT constructor
    if opt.use_ema:
        ema = deepcopy(model).to(
            gpu
        )  # Create an EMA of the model for use after training
        requires_grad(ema, False)

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

    print("Model = %s" % str(model))
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    print("Total_params = %s MB " % str(total_params))

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0)

    if opt.model != "":
        if opt.distribution_type == "multi":
            ckpt = torch.load(opt.model, map_location="cuda:%d" % gpu)
        else:
            ckpt = torch.load(opt.model)
        if opt.model.endswith("expand_from_4096.pth"):
            model.load_state_dict(ckpt["model_state"])
            # optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
        else:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
        # modify the lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt.lr  # use the new lr

        del ckpt

    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    def new_y_chain(y, num_chain, num_classes):
        return torch.randint(
            low=0, high=num_classes, size=(num_chain,), device=y.device
        )

    # Prepare models for training:
    if opt.use_ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode

    # detect anomaly torch
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, opt.niter):
        if opt.distribution_type == "multi":
            train_sampler.set_epoch(epoch)

        for idx, data in enumerate(dataloader):
            ref_pts = data["face_pts"].transpose(1, 2) * 2.0  # [-1, 1]
            x = data["gt_pos"].transpose(1, 2) * 2.0

            # x = ref_pts.clone() # try if this is working
            noises_batch = noises_init[data["idx"]].transpose(1, 2)
            y = data["cate_idx"]

            # """
            # DEBUG: add noise normal?
            # """
            # x_gen_addnoise = model.gen_addnoise_traj(
            #     data_start=x[:1],
            #     device=x.device,
            #     freq=40,
            # )
            # x_gen_addnoise = torch.cat(x_gen_addnoise, dim=0)

            # visualize_pointcloud_batch(
            #     "%s/addnoise_traj.png" % (outf_syn),
            #     x_gen_addnoise.transpose(1, 2),
            #     None,
            #     None,
            #     None,
            #     discrete=False,
            # )
            # a = 1

            """
            train diffusion
            """

            if opt.distribution_type == "multi" or (
                opt.distribution_type is None and gpu is not None
            ):
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
                y = y.cuda(gpu)
                ref_pts = ref_pts.cuda(gpu)
            elif opt.distribution_type == "single":
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                y = y.cuda()
                ref_pts = ref_pts.cuda()

            if opt.pe_output:
                x = model.pe(x)
            loss = model.get_loss_iter(x, ref_pts, noises_batch, y).mean()

            optimizer.zero_grad()
            loss.backward()
            # netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()

            if opt.use_ema:
                update_ema(ema, model)

            if not opt.debug:
                global_step = idx + len(dataloader) * epoch
                if opt.use_tb:
                    tb_writer.add_scalar("train_loss", loss.item(), global_step)
                    tb_writer.add_scalar(
                        "train_lr", optimizer.param_groups[0]["lr"], global_step
                    )
                if opt.use_wandb and gpu == 1:
                    wandb.log(
                        {
                            "train/train_loss": loss.item(),
                            "train/train_lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

            if idx % opt.print_freq == 0 and should_diag:
                logger.info(
                    "[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    ".format(
                        epoch, opt.niter, idx, len(dataloader), loss.item()
                    )
                )

        if (epoch + 1) % opt.diagIter == 0 and should_diag:
            logger.info("Diagnosis:")

            x_range = [x.min().item(), x.max().item()]
            kl_stats = model.all_kl(x, ref_pts, y)
            logger.info(
                "      [{:>3d}/{:>3d}]    "
                "x_range: [{:>10.4f}, {:>10.4f}],   "
                "total_bpd_b: {:>10.4f},    "
                "terms_bpd: {:>10.4f},  "
                "prior_bpd_b: {:>10.4f}    "
                "mse_bt: {:>10.4f}  ".format(
                    epoch,
                    opt.niter,
                    *x_range,
                    kl_stats["total_bpd_b"].item(),
                    kl_stats["terms_bpd"].item(),
                    kl_stats["prior_bpd_b"].item(),
                    kl_stats["mse_bt"].item(),
                )
            )

            if not opt.debug:
                if opt.use_tb:
                    tb_writer.add_scalar(
                        "total_bpd_b", kl_stats["total_bpd_b"].item(), epoch
                    )
                    tb_writer.add_scalar(
                        "terms_bpd", kl_stats["terms_bpd"].item(), epoch
                    )
                    tb_writer.add_scalar(
                        "prior_bpd_b", kl_stats["prior_bpd_b"].item(), epoch
                    )
                    tb_writer.add_scalar("mse_bt", kl_stats["mse_bt"].item(), epoch)
                if opt.use_wandb and gpu == 1:
                    wandb.log(
                        {
                            "diag/total_bpd_b": kl_stats["total_bpd_b"].item(),
                            "diag/terms_bpd": kl_stats["terms_bpd"].item(),
                            "diag/prior_bpd_b": kl_stats["prior_bpd_b"].item(),
                            "diag/mse_bt": kl_stats["mse_bt"].item(),
                        },
                        step=epoch,
                    )

        if ((epoch + 1) % opt.vizIter) == 0 and gpu == 1:
            logger.info("Generation: eval")

            model.eval()
            random_data = get_random_batch(test_dataloader)
            ref_pts = random_data["face_pts"].transpose(1, 2) * 2.0
            x = random_data["gt_pos"].transpose(1, 2) * 2.0
            noises_batch = noises_init[random_data["idx"]].transpose(1, 2)
            y = random_data["cate_idx"]

            if opt.distribution_type == "multi" or (
                opt.distribution_type is None and gpu is not None
            ):
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
                y = y.cuda(gpu)
                ref_pts = ref_pts.cuda(gpu)
            elif opt.distribution_type == "single":
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                y = y.cuda()
                ref_pts = ref_pts.cuda()
            # with torch.no_grad():

            with torch.no_grad():
                eval_loss = model.get_loss_iter(x, ref_pts, noises_batch, y).mean()

                if opt.use_wandb:
                    wandb.log(
                        {
                            "eval/eval_loss": eval_loss.item(),
                        },
                        step=global_step,
                    )
                visualize_pointcloud_batch(
                    "%s/epoch_%03d_x.png" % (outf_syn, epoch),
                    x.transpose(1, 2),
                    None,
                    None,
                    None,
                )

                x_gen_eval = model.gen_samples(
                    new_x_chain(x, len(x)).shape,
                    ref_pts,
                    None,
                    x.device,
                    y,
                    clip_denoised=False,
                )

                visualize_pointcloud_batch(
                    "%s/epoch_%03d_samples_eval.png" % (outf_syn, epoch),
                    x_gen_eval.transpose(1, 2)[:, :, :3],
                    None,
                    None,
                    None,
                    discrete=False,
                )

                visualize_pointcloud_batch(
                    "%s/epoch_%03d_samples_eval_discrete.png" % (outf_syn, epoch),
                    x_gen_eval.transpose(1, 2)[:, :, :3],
                    None,
                    None,
                    None,
                    discrete=True,
                )

                x_gen_list = model.gen_sample_traj(
                    new_x_chain(x, 1).shape,
                    ref_pts[:1],
                    None,
                    x.device,
                    y[:1],
                    freq=40,
                    clip_denoised=False,
                )
                x_gen_all = torch.cat(x_gen_list, dim=0)

                visualize_pointcloud_batch(
                    "%s/epoch_%03d_samples_eval_all.png" % (outf_syn, epoch),
                    x_gen_all.transpose(1, 2)[:, :, :3],
                    None,
                    None,
                    None,
                )

                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                logger.info(
                    "      [{:>3d}/{:>3d}]  "
                    "eval_gen_range: [{:>10.4f}, {:>10.4f}]     "
                    "eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      ".format(
                        epoch,
                        opt.niter,
                        *gen_eval_range,
                        *gen_stats,
                    )
                )

            logger.info("Generation: train")
            # delete variables to save memory
            del x_gen_eval, x_gen_list, x_gen_all, x, ref_pts, noises_batch, y
            model.train()

        # if (epoch + 1) % opt.evalIter == 0 and gpu == 1:
        #     logger.info("Loss: eval")

        #     model.eval()
        #     random_data = get_random_batch(test_dataloader)
        #     ref_pts = random_data["face_pts"].transpose(1, 2) * 2.0  # [-1, 1]
        #     x = random_data["gt_pos"].transpose(1, 2) * 2.0
        #     noises_batch = noises_init[random_data["idx"]].transpose(1, 2)
        #     y = random_data["cate_idx"]

        #     if opt.distribution_type == "multi" or (
        #         opt.distribution_type is None and gpu is not None
        #     ):
        #         x = x.cuda(gpu)
        #         noises_batch = noises_batch.cuda(gpu)
        #         y = y.cuda(gpu)
        #         ref_pts = ref_pts.cuda(gpu)
        #     elif opt.distribution_type == "single":
        #         x = x.cuda()
        #         noises_batch = noises_batch.cuda()
        #         y = y.cuda()
        #         ref_pts = ref_pts.cuda()

        #     eval_loss = model.get_loss_iter(x, ref_pts, noises_batch, y).mean()

        #     if opt.use_wandb:
        #         wandb.log(
        #             {
        #                 "eval/eval_loss": eval_loss.item(),
        #             },
        #             step=global_step,
        #         )
        #     model.train()

        if (epoch + 1) % opt.saveIter == 0:
            if should_diag:
                save_dict = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }

                if opt.use_ema:
                    save_dict.update({"ema": ema.state_dict()})

                torch.save(save_dict, "%s/epoch_%d.pth" % (output_dir, epoch))

            if opt.distribution_type == "multi":
                dist.barrier()
                map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
                if opt.use_ema:
                    checkpoint = torch.load(
                        "%s/epoch_%d.pth" % (output_dir, epoch),
                        map_location=map_location,
                    )["ema"]
                    checkpoint_dict = {
                        k.replace("model.", "model.module."): checkpoint[k]
                        for k in checkpoint
                        if k.startswith("model.")
                    }
                    model.load_state_dict(checkpoint_dict)
                else:
                    model.load_state_dict(
                        torch.load(
                            "%s/epoch_%d.pth" % (output_dir, epoch),
                            map_location=map_location,
                        )["model_state"]
                    )

    dist.destroy_process_group()


def check_betas(betas_start, betas_end, schedule_type, time_num):
    betas = get_betas(schedule_type, betas_start, betas_end, time_num)

    # multiply all 1 - II(1 - beta_t) to get the final data
    mult = 1.0  # alpha_bar
    for beta_t in betas:
        mult *= 1 - beta_t

    var_T = np.sqrt(1 - mult)
    alpha_t = 1 - beta_t
    mu_xt_t = (1 / np.sqrt(alpha_t)) * (beta_t / np.sqrt(1 - mult))

    return


def main():
    opt = parse_args()
    opt.beta_start = 3e-7  # 1e-7
    opt.beta_end = 0.1  # 0.3
    opt.schedule_type = "exp"  # "warm0.5"
    opt.model_mean_type = "xstart"

    # check_betas(opt.beta_start, opt.beta_end, opt.schedule_type, opt.time_num)

    output_dir = get_output_dir(opt.model_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    """ workaround """
    train_dataset, val_dataset = get_dataset(opt, opt.npoints, opt.category)
    if opt.pe_output:
        noises_init = torch.randn(len(train_dataset), opt.npoints, 51)
    else:
        noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    # noises_init = torch.randn(1, opt.npoints, opt.nc).repeat(len(train_dataset), 1, 1)
    # noises_init = torch.clamp(noises_init, -1.0, 1.0) # make sure the scale is ok

    # Use random port to avoid collision between parallel jobs
    if opt.world_size == 1:
        opt.port = np.random.randint(10000, 20000)
    opt.dist_url = f"tcp://{opt.node}:{opt.port}"
    print("Using url {}".format(opt.dist_url))

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)


if __name__ == "__main__":
    main()
