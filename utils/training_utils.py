import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def softargmax(logits):
    smax = torch.nn.functional.softmax(logits, dim=-1)
    tokens = (
        torch.tensor(
            list(range(logits.shape[-1])), dtype=torch.float32, device=logits.device
        )
        .reshape(1, 1, -1)
        .expand_as(smax)
    )
    return torch.sum(tokens * smax, dim=-1)


class PE(nn.Module):
    def __init__(self, hidden_dim=48):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        input = input.transpose(1, 2)
        # input: B x N x 3
        emb = self.embed(input, self.basis)
        embed = torch.cat([input, emb], dim=2)  # B x C x N

        return embed.transpose(1, 2)


def norm(v, f):
    v = (v - v.min()) / (v.max() - v.min()) - 0.5

    return v, f


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)


def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p**2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad**2) for p in net.parameters()))
    return pNorm, gradNorm


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if name.startswith("model.module"):
            name = name.replace("model.module.", "model.")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./checkpoints",
        help="path to save trained model weights",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="debug",
        help="experiment name (used for checkpointing and logging)",
    )

    # Data params
    # parser.add_argument("--dataroot", default="ShapeNetCore.v2.PC15k/")
    parser.add_argument("--category", default=None)
    parser.add_argument("--num_classes", type=int, default=55)

    parser.add_argument("--bs", type=int, default=16, help="input batch size")
    parser.add_argument("--workers", type=int, default=16, help="workers")
    parser.add_argument(
        "--niter", type=int, default=10000, help="number of epochs to train for"
    )

    parser.add_argument("--nc", default=3)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument(
        "--voxel_size", type=int, choices=[16, 32, 64, 128, 256], default=32
    )

    """model"""
    parser.add_argument("--model_type", type=str, default="DiT-B/4")
    parser.add_argument("--beta_start", default=0.0001)
    parser.add_argument("--beta_end", default=0.02)
    parser.add_argument("--schedule_type", default="linear")
    parser.add_argument("--time_num", type=int, default=1000)

    # params
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--window_block_indexes", type=tuple, default="0,3,6,9")
    parser.add_argument("--attention", default=True)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--loss_type", default="mse")
    parser.add_argument("--model_mean_type", default="eps")
    parser.add_argument("--model_var_type", default="fixedsmall")

    parser.add_argument(
        "--lr", type=float, default=2e-4, help="learning rate for E, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--decay", type=float, default=0, help="weight decay for EBM")
    parser.add_argument(
        "--grad_clip", type=float, default=None, help="weight decay for EBM"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.998, help="lr decay for EBM"
    )

    parser.add_argument(
        "--model", default="", help="path to model (to continue training)"
    )

    """distributed"""
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed nodes."
    )
    parser.add_argument("--node", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--dist_url", type=str, default="tcp://localhost:12345")
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--distribution_type",
        default="single",
        choices=["multi", "single", None],
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="GPU id to use. None means using all available GPUs.",
    )

    """eval"""
    parser.add_argument("--saveIter", default=1, type=int, help="unit: epoch")
    parser.add_argument("--diagIter", default=50000, type=int, help="unit: epoch")
    parser.add_argument("--vizIter", default=1, type=int, help="unit: epoch")
    parser.add_argument("--evalIter", default=1, type=int, help="unit: epoch")
    parser.add_argument("--print_freq", default=50, type=int, help="unit: iter")

    parser.add_argument("--manualSeed", default=42, type=int, help="random seed")

    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug mode"
    )
    parser.add_argument(
        "--use_tb", action="store_true", default=False, help="use tensorboard"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help="use wandb"
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=False,
        help="use pretrained 2d DiT weights",
    )
    parser.add_argument("--use_ema", action="store_true", default=False, help="use ema")

    """
    meshgpt configs, moved here
    """
    # boolean flags
    parser.add_argument(
        "--scale_augment", action="store_true", default=False, help="scale augment"
    )
    parser.add_argument(
        "--shift_augment", action="store_true", default=False, help="shift augment"
    )
    parser.add_argument(
        "--rot_augment", action="store_true", default=False, help="rotation augment"
    )
    parser.add_argument(
        "--noise_augment", action="store_true", default=False, help="noise augment"
    )
    parser.add_argument(
        "--ce_output", action="store_true", default=True, help="ce output"
    )
    parser.add_argument(
        "--pe_output", action="store_true", default=False, help="pe output"
    )
    parser.add_argument(
        "--low_augment", action="store_true", default=False, help="low augment"
    )

    # int values
    parser.add_argument(
        "--num_tokens", default=131, type=int, help="num quantize tokens"
    )
    parser.add_argument("--pad_to_faces", default=800, type=int, help="num faces")
    parser.add_argument("--start_idx", default=0, type=int, help="num faces")

    # paths
    parser.add_argument("--dataroot", default="data/shapenet/processed_data.pkl")

    opt = parser.parse_args()

    return opt
