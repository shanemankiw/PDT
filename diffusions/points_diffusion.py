import numpy as np
import torch
from torch.distributions import Normal
from tqdm import tqdm


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


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == "linear":
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == "warm0.1":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.2":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.5":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "exp":
        t = np.linspace(0, 1, time_num)
        betas = b_start * (b_end / b_start) ** t
    elif schedule_type == "lin_exp":
        betas = np.zeros(time_num)
        # int stage
        int_step = 200
        int_end = 2 * b_start
        # climb stage
        climb_step = 800
        climb_end = b_start * 1000
        # exp stage
        exp_end = b_end

        for i in range(int_step):
            betas[i] = b_start + (int_end - b_start) * (i / int_step)
        for i in range(int_step, climb_step):
            betas[i] = int_end + (climb_end - int_end) * (i - int_step) / (
                climb_step - int_step
            )
        for i in range(climb_step, time_num):
            betas[i] = climb_end * (exp_end / climb_end) ** (
                (i - climb_step) / (time_num - climb_step)
            )
        betas = np.clip(betas, b_start, b_end)
    elif schedule_type == "lin_exp_1":
        betas = np.zeros(time_num)
        # int stage
        int_step = 200
        int_end = b_start
        # climb stage
        climb_step = 800
        climb_end = 10 * b_start
        # exp stage
        exp_end = b_end

        for i in range(int_step):
            betas[i] = b_start + (int_end - b_start) * (i / int_step)
        for i in range(int_step, climb_step):
            betas[i] = int_end + (climb_end - int_end) * (i - int_step) / (
                climb_step - int_step
            )
        for i in range(climb_step, time_num):
            betas[i] = climb_end * (exp_end / climb_end) ** (
                (i - climb_step) / (time_num - climb_step)
            )
        betas = np.clip(betas, b_start, b_end)
    elif schedule_type == "cosine":
        return betas_for_alpha_bar(
            time_num,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            max_beta=b_end,
        )
    else:
        raise NotImplementedError(schedule_type)
    return betas


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(
            np.float64
        )  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        betas = np.concatenate(
            [betas, np.full_like(betas[: int(0.3 * len(betas))], betas[-1])]
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(
            np.append(1.0, alphas_cumprod[:-1])
        ).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
        )
        variance = self._extract(
            1.0 - self.alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
            )
            * noise
        )

    def get_noise_weights(self, tb, device):
        return 1.0 / self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(device), tb, tb.shape
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape)
            * x_start
            + self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape)
            * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance.to(x_start.device), t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        denoise_fn,
        data,
        data_ref,
        cond,
        t,
        y,
        clip_denoised: bool,
        return_pred_xstart: bool,
    ):
        # print('data:', data.shape)
        model_intput = torch.cat([data, data_ref], dim=1)
        model_output = denoise_fn(model_intput, cond, t, y)
        # print('model_output:', model_output.shape)

        if self.model_var_type in ["fixedsmall", "fixedlarge"]:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                "fixedlarge": (
                    self.betas.to(data.device),
                    torch.log(
                        torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                    ).to(data.device),
                ),
                "fixedsmall": (
                    self.posterior_variance.to(data.device),
                    self.posterior_log_variance_clipped.to(data.device),
                ),
            }[self.model_var_type]
            model_variance = self._extract(
                model_variance, t, data.shape
            ) * torch.ones_like(data)
            model_log_variance = self._extract(
                model_log_variance, t, data.shape
            ) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == "eps":
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -0.5, 0.5)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x_recon, x_t=data, t=t
            )
        elif self.model_mean_type == "xstart":
            x_recon = model_output
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x_recon, x_t=data, t=t
            )
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape)
            * x_t
            - self._extract(
                self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape
            )
            * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, x_start):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape)
            * x_t
            - x_start
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)

    """ samples """

    def p_sample(
        self,
        denoise_fn,
        data,
        data_ref,
        cond,
        t,
        noise_fn,
        y,
        clip_denoised=False,
        return_pred_xstart=False,
    ):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            data=data,
            data_ref=data_ref,
            cond=cond,
            t=t,
            y=y,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(
            1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1)
        )

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(
        self,
        denoise_fn,
        shape,
        data_ref,
        cond,
        device,
        y,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
    ):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in tqdm(
            reversed(
                range(0, self.num_timesteps if not keep_running else len(self.betas))
            )
        ):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                cond=cond,
                data=img_t,
                data_ref=data_ref,
                t=t_,
                noise_fn=noise_fn,
                y=y,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            )

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_debug(
        self,
        denoise_fn,
        x_start,
        data_ref,
        cond,
        device,
        y,
        t_max=300,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
    ):
        """
        DEBUG: given gt, sample it for several steps and then try to denoise it back.
        """
        B = x_start.shape[0]
        t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t_max)
        noise = torch.randn(x_start.shape, device=x_start.device)

        x_t = self.q_sample(x_start=x_start, t=t_b, noise=noise)
        for t in tqdm(reversed(range(0, t_max))):
            t_ = torch.empty(B, dtype=torch.int64, device=device).fill_(t)
            x_t = self.p_sample(
                denoise_fn=denoise_fn,
                cond=cond,
                data=x_t,
                data_ref=data_ref,
                t=t_,
                noise_fn=noise_fn,
                y=y,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            )

        assert x_t.shape == x_start.shape
        return x_t

    def p_sample_loop_wgt(
        self,
        denoise_fn,
        shape,
        data_ref,
        cond,
        gt,
        device,
        y,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
    ):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        # img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        img_t = self.q_sample(
            x_start=gt,
            t=torch.empty(shape[0], dtype=torch.int64, device=device).fill_(
                self.num_timesteps - 1
            ),
        )
        for t in tqdm(
            reversed(
                range(0, self.num_timesteps if not keep_running else len(self.betas))
            )
        ):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                cond=cond,
                data=img_t,
                data_ref=data_ref,
                t=t_,
                noise_fn=noise_fn,
                y=y,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            )

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(
        self,
        denoise_fn,
        cond,
        shape,
        data_ref,
        device,
        y,
        freq,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
    ):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps = self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in tqdm(reversed(range(0, total_steps))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                cond=cond,
                data_ref=data_ref,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                y=y,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            )
            if t % freq == 0 or t == total_steps - 1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    def q_sample_loop_trajectory(
        self,
        data_start,
        device,
        freq,
        noise_fn=torch.randn,
    ):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        B, D, N = data_start.shape

        total_steps = len(self.betas)

        noise = noise_fn(
            data_start.shape, dtype=data_start.dtype, device=data_start.device
        )
        pcs = []
        for t in tqdm(range(0, total_steps)):
            t_ = torch.empty([B], dtype=torch.int64, device=device).fill_(t)
            data_t = self.q_sample(x_start=data_start, t=t_, noise=noise)
            if t % freq == 0 or t == total_steps - 1:
                pcs.append(data_t)

        assert pcs[-1].shape == data_start.shape

        return pcs

    """losses"""

    def _vb_terms_bpd(
        self,
        denoise_fn,
        data_start,
        data_ref,
        data_t,
        cond,
        t,
        y,
        clip_denoised: bool,
        return_pred_xstart: bool,
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start, x_t=data_t, t=t
        )
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            data=data_t,
            data_ref=data_ref,
            cond=cond,
            t=t,
            y=y,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.0)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(
        self,
        denoise_fn,
        data_start,
        data_ref,
        cond,
        t,
        noise=None,
        y=None,
        aux_weight=1e-4,
    ):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(
                data_start.shape, dtype=data_start.dtype, device=data_start.device
            )
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == "mse":
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            data_t_cat = torch.cat([data_t, data_ref], dim=1)
            # eps_recon = denoise_fn(data_t_cat, cond, t, y)
            # assert data_t.shape == data_start.shape
            # assert eps_recon.shape == torch.Size([B, D, N])
            # assert eps_recon.shape == data_start.shape
            if self.model_mean_type == "eps":
                eps_recon = denoise_fn(data_t_cat, cond, t, y)
                losses = ((noise - eps_recon) ** 2).mean(
                    dim=list(range(1, len(data_start.shape)))
                )
                if aux_weight > 0:
                    pred_start = self._predict_xstart_from_eps(data_t, t, eps_recon)
                    aux_loss = aux_weight * ((data_start - pred_start) ** 2).mean(
                        dim=list(range(1, len(data_start.shape)))
                    )
                    losses += aux_loss

            elif self.model_mean_type == "xstart":
                x_start_recon = denoise_fn(data_t_cat, cond, t, y)
                losses = ((data_start.detach() - x_start_recon) ** 2).mean(
                    dim=list(range(1, len(data_start.shape)))
                )  # predict x_0 instead of eps
        elif self.loss_type == "kl":
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn,
                data_start=data_start,
                data_ref=data_ref,
                data_t=data_t,
                cond=cond,
                t=t,
                y=y,
                clip_denoised=False,
                return_pred_xstart=False,
            )
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    """debug"""

    def _prior_bpd(self, x_start):
        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T - 1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(
                mean1=qt_mean,
                logvar1=qt_log_variance,
                mean2=torch.tensor([0.0]).to(qt_mean),
                logvar2=torch.tensor([0.0]).to(qt_log_variance),
            )
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.0)

    def calc_bpd_loop(self, denoise_fn, data_ref, cond, x_start, y, clip_denoised=True):
        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_, eps_mse_bt_ = (
                torch.zeros([B, T], device=x_start.device),
                torch.zeros([B, T], device=x_start.device),
                torch.zeros([B, T], device=x_start.device),
            )
            for t in tqdm(reversed(range(T))):
                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                noise = torch.randn(x_start.shape, device=x_start.device)
                x_t = self.q_sample(x_start=x_start, t=t_b, noise=noise)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn,
                    data_start=x_start,
                    data_ref=data_ref,
                    data_t=x_t,
                    cond=cond,
                    t=t_b,
                    y=y,
                    clip_denoised=clip_denoised,
                    return_pred_xstart=True,
                )
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart - x_start) ** 2).mean(
                    dim=list(range(1, len(x_start.shape)))
                )

                # MSE for eps mse
                eps = self._predict_eps_from_xstart(x_t, t_b, x_start)
                eps_mse_b = ((eps - noise) ** 2).mean(
                    dim=list(range(1, len(noise.shape)))
                )

                assert new_vals_b.shape == new_mse_b.shape == torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = (
                    t_b[:, None] == torch.arange(T, device=t_b.device)[None, :].float()
                )
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                eps_mse_bt_ = eps_mse_bt_ * (~mask_bt) + eps_mse_b[:, None] * mask_bt
                assert (
                    mask_bt.shape
                    == vals_bt_.shape
                    == vals_bt_.shape
                    == torch.Size([B, T])
                )

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size(
                [B, T]
            ) and total_bpd_b.shape == prior_bpd_b.shape == torch.Size([B])
            return (
                total_bpd_b,  # .mean()
                vals_bt_,
                prior_bpd_b,
                mse_bt_,
                eps_mse_bt_,
            )
