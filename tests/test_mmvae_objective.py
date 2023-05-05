"""Test functions in the MMVAE model"""

import math

import numpy as np
import pytest
import torch
import torch.distributions as dist

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models import MMVAE, MMVAEConfig


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def compute_microbatch_split(x, K):
    """Checks if batch needs to be broken down further to fit in memory."""
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = (
        sum([1.0 / (K * math.prod(_x.size()[1:])) for _x in x])
        if is_multidata(x)
        else 1.0 / (K * math.prod(x.size()[1:]))
    )
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert S > 0, "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def _m_dreg_looser(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(
            torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_])
        )
        lpx_z = [
            px_z.log_prob(x[d])
            .view(*px_z.batch_shape[:2], -1)
            .mul(model.vaes[d].llik_scaling)
            .sum(-1)
            for d, px_z in enumerate(px_zs[r])
        ]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss)


def _m_dreg_looser_test(qz_xs_, px_zs, zss, model: MMVAE, x, rescale_factors, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    lws = []
    for r, vae in enumerate(model.encoders):
        lpz = model.prior_dist(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(
            torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_])
        )
        lpx_z = [
            px_z.log_prob(x[d])
            .view(*px_z.batch_shape[:2], -1)
            .mul(rescale_factors[d])
            .sum(-1)
            for d, px_z in enumerate(px_zs[r])
        ]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    lw = torch.stack(lws)
    zss = torch.stack(zss)
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return -(grad_wt * lw).mean(0).sum()


def _m_iwae_test(qz_xs, px_zs, zss, model: MMVAE, x, rescale_factors, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.prior_dist(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(
            torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs])
        )
        lpx_z = [
            px_z.log_prob(x[d])
            .view(*px_z.batch_shape[:2], -1)
            .mul(rescale_factors[d])
            .sum(-1)
            for d, px_z in enumerate(px_zs[r])
        ]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x

        lws.append(lw)
    lws = torch.cat(lws)  # (n_modality * n_samples) x batch_size

    return log_mean_exp(lws, dim=0).sum()


def m_dreg_looser(model, x, K=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    zss = torch.cat(zss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()


class Test_mmvae_obj:
    @pytest.fixture
    def dataset(self):
        # Create simple small dataset
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
        )
        labels = np.array([0, 1])
        dataset = MultimodalBaseDataset(data, labels)
        return dataset

    @pytest.fixture(params=[True, False])
    def model_config(self, request):
        model_config = MMVAEConfig(
            n_modalities=2,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,)),
            use_likelihood_rescaling=request.param,
            decoder_dist_params=dict(mod1=dict(scale=0.75), mod2=dict(scale=0.75)),
            decoders_dist=dict(mod1="laplace", mod2="laplace"),
            K=5,
        )

        return model_config

    @pytest.fixture
    def model(self, model_config):
        model = MMVAE(model_config)
        return model

    def test_forward(self, model, dataset):
        out = model(dataset, detailed_output=True)

        assert hasattr(out, "qz_xs_detach")

        qz_xs_detach_dict = out.qz_xs_detach
        qz_xs_dict = out.qz_xs

        zss_dict = out.zss
        px_zs_dict = out.recon

        qz_xs_detach = []
        qz_xs = []
        zss = []
        px_zs = []
        keys = list(zss_dict.keys())
        for i, mod in enumerate(keys):
            zss.append(zss_dict[mod])
            qz_xs_detach.append(qz_xs_detach_dict[mod])
            qz_xs.append(qz_xs_dict[mod])
            px_zs.append([])
            for j, r_mod in enumerate(keys):
                recon = px_zs_dict[mod][r_mod]

                px_zs[-1].append(dist.Laplace(recon, 0.75))

        x = [dataset.data[mod] for mod in keys]
        rescale_factors = [model.rescale_factors[mod] for mod in keys]
        test_loss = _m_dreg_looser_test(
            qz_xs_detach, px_zs, zss, model, x, rescale_factors, K=model.K
        )

        assert test_loss == out.loss

        test_iwae = -_m_iwae_test(
            qz_xs, px_zs, zss, model, x, rescale_factors, K=model.K
        )
        model_iwae = model.iwae(qz_xs_dict, zss_dict, out.recon, dataset).loss
        assert test_iwae == model_iwae
