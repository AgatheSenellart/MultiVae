import argparse
import os
from abc import ABC, abstractmethod
from itertools import chain, combinations

import torch
import torch.nn as nn
from torch.autograd import Variable


def poe(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    T = 1.0 / var
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1.0 / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar


def reweight_weights(w):
    w = w / w.sum()
    return w


def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    # if not defined, take pre-defined weights
    num_components = mus.shape[0]
    num_samples = mus.shape[1]
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device)
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0
        else:
            i_start = int(idx_end[k - 1])
        if k == w_modalities.shape[0] - 1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples
    mu_sel = torch.cat(
        [mus[k, idx_start[k] : idx_end[k], :] for k in range(w_modalities.shape[0])]
    )
    logvar_sel = torch.cat(
        [logvars[k, idx_start[k] : idx_end[k], :] for k in range(w_modalities.shape[0])]
    )
    return [mu_sel, logvar_sel]


def calc_kl_divergence(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
    if mu1 is None or logvar1 is None:
        KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
    else:
        KLD = -0.5 * (
            torch.sum(
                1
                - logvar0.exp() / logvar1.exp()
                - (mu0 - mu1).pow(2) / logvar1.exp()
                + logvar0
                - logvar1
            )
        )
    if norm_value is not None:
        KLD = KLD / float(norm_value)
    return KLD


def calc_group_divergence_moe(flags, mus, logvars, weights, normalization=None):
    num_mods = mus.shape[0]
    num_samples = mus.shape[1]
    if normalization is not None:
        klds = torch.zeros(num_mods)
    else:
        klds = torch.zeros(num_mods, num_samples)
    klds = klds.to(flags.device)
    weights = weights.to(flags.device)
    for k in range(0, num_mods):
        kld_ind = calc_kl_divergence(
            mus[k, :, :], logvars[k, :, :], norm_value=normalization
        )
        if normalization is not None:
            klds[k] = kld_ind
        else:
            klds[k, :] = kld_ind
    if normalization is None:
        weights = weights.unsqueeze(1).repeat(1, num_samples)
    group_div = (weights * klds).sum(dim=0)
    return group_div, klds


class BaseMMVae(ABC, nn.Module):
    def __init__(self, flags, modalities):
        super(BaseMMVae, self).__init__()
        self.num_modalities = len(modalities.keys())
        self.flags = flags
        self.modalities = modalities
        self.subsets = self.set_subsets()
        self.set_fusion_functions()

        encoders = nn.ModuleDict()
        decoders = nn.ModuleDict()
        lhoods = dict()
        for m, m_key in enumerate(sorted(modalities.keys())):
            encoders[m_key] = modalities[m_key].encoder
            decoders[m_key] = modalities[m_key].decoder
            lhoods[m_key] = modalities[m_key].likelihood
        self.encoders = encoders
        self.decoders = decoders
        self.lhoods = lhoods

    def set_subsets(self):
        num_mods = len(list(self.modalities.keys()))

        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        xs = list(self.modalities)
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(
            combinations(xs, n) for n in range(len(xs) + 1)
        )
        subsets = dict()
        for k, mod_names in enumerate(subsets_list):
            mods = []
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.modalities[mod_name])
            key = "_".join(sorted(mod_names))
            subsets[key] = mods
        return subsets

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def set_fusion_functions(self):
        self.modality_fusion = self.poe_fusion
        self.fusion_condition = self.fusion_condition_joint
        self.calc_joint_divergence = self.divergence_static_prior

    def divergence_static_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = weights.clone()
        weights = reweight_weights(weights)
        div_measures = calc_group_divergence_moe(
            self.flags, mus, logvars, weights, normalization=self.flags.batch_size
        )
        divs = dict()
        divs["joint_divergence"] = div_measures[0]
        divs["individual_divs"] = div_measures[1]
        divs["dyn_prior"] = None
        return divs

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = reweight_weights(weights)
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        mu_moe, logvar_moe = mixture_component_selection(
            self.flags, mus, logvars, weights
        )
        return [mu_moe, logvar_moe]

    def poe_fusion(self, mus, logvars, weights=None):
        if self.flags.modality_poe or mus.shape[0] == len(self.modalities.keys()):
            num_samples = mus[0].shape[0]
            mus = torch.cat(
                (
                    mus,
                    torch.zeros(1, num_samples, self.flags.class_dim).to(
                        self.flags.device
                    ),
                ),
                dim=0,
            )
            logvars = torch.cat(
                (
                    logvars,
                    torch.zeros(1, num_samples, self.flags.class_dim).to(
                        self.flags.device
                    ),
                ),
                dim=0,
            )
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        mu_poe, logvar_poe = poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def fusion_condition_moe(self, subset, input_batch=None):
        if len(subset) == 1:
            return True
        else:
            return False

    def fusion_condition_poe(self, subset, input_batch=None):
        if len(subset) == len(input_batch.keys()):
            return True
        else:
            return False

    def fusion_condition_joint(self, subset, input_batch=None):
        return True

    def forward(self, input_batch):
        latents = self.inference(input_batch)
        results = dict()
        results["latents"] = latents
        results["group_distr"] = latents["joint"]
        class_embeddings = self.reparameterize(latents["joint"][0], latents["joint"][1])
        div = self.calc_joint_divergence(
            latents["mus"], latents["logvars"], latents["weights"]
        )
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        results_rec = dict()
        enc_mods = latents["modalities"]
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                m_s_mu, m_s_logvar = enc_mods[m_key + "_style"]
                if self.flags.factorized_representation:
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar)
                else:
                    m_s_embeddings = None
                m_rec = self.lhoods[m_key](
                    self.decoders[m_key](class_embeddings).reconstruction, 0.75
                )
                results_rec[m_key] = m_rec
        results["rec"] = results_rec
        return results

    def encode(self, input_batch):
        latents = dict()
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                i_m = input_batch[m_key]
                l = self.encoders[m_key](i_m)
                latents[m_key + "_style"] = [None, None]
                latents[m_key] = [l.embedding, l.log_covariance]
            else:
                latents[m_key + "_style"] = [None, None]
                latents[m_key] = [None, None]
        return latents

    def inference(self, input_batch, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size
        latents = dict()
        enc_mods = self.encode(input_batch)
        latents["modalities"] = enc_mods
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)
        distr_subsets = dict()
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != "":
                mods = self.subsets[s_key]
                mus_subset = torch.Tensor().to(self.flags.device)
                logvars_subset = torch.Tensor().to(self.flags.device)
                mods_avail = True
                for m, mod in enumerate(mods):
                    if mod.name in input_batch.keys():
                        mus_subset = torch.cat(
                            (mus_subset, enc_mods[mod.name][0].unsqueeze(0)), dim=0
                        )
                        logvars_subset = torch.cat(
                            (logvars_subset, enc_mods[mod.name][1].unsqueeze(0)), dim=0
                        )
                    else:
                        mods_avail = False
                if mods_avail:
                    weights_subset = (1 / float(len(mus_subset))) * torch.ones(
                        len(mus_subset)
                    ).to(self.flags.device)
                    s_mu, s_logvar = self.modality_fusion(
                        mus_subset, logvars_subset, weights_subset
                    )
                    distr_subsets[s_key] = [s_mu, s_logvar]
                    if self.fusion_condition(mods, input_batch):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)

        # weights = (1/float(len(mus)))*torch.ones(len(mus)).to(self.flags.device);
        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(
            self.flags.device
        )
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        latents["mus"] = mus
        latents["logvars"] = logvars
        latents["weights"] = weights
        latents["joint"] = [joint_mu, joint_logvar]
        latents["subsets"] = distr_subsets
        return latents

    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        mu = torch.zeros(num_samples, self.flags.class_dim).to(self.flags.device)
        logvar = torch.zeros(num_samples, self.flags.class_dim).to(self.flags.device)
        z_class = self.reparameterize(mu, logvar)
        z_styles = self.get_random_styles(num_samples)
        random_latents = {"content": z_class, "style": z_styles}
        random_samples = self.generate_from_latents(random_latents)
        return random_samples

    def generate_sufficient_statistics_from_latents(self, latents):
        suff_stats = dict()
        content = latents["content"]
        for m, m_key in enumerate(self.modalities.keys()):
            s = latents["style"][m_key]
            cg = self.lhoods[m_key](*self.decoders[m_key](s, content))
            suff_stats[m_key] = cg
        return suff_stats

    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents)
        cond_gen = dict()
        for m, m_key in enumerate(latents["style"].keys()):
            cond_gen_m = suff_stats[m_key].mean
            cond_gen[m_key] = cond_gen_m
        return cond_gen

    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_samples = dict()
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key]
            content_rep = self.reparameterize(mu=mu, logvar=logvar)
            latents = {"content": content_rep, "style": style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents)
        return cond_gen_samples

    def get_random_style_dists(self, num_samples):
        styles = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            s_mu = torch.zeros(num_samples, mod.style_dim).to(self.flags.device)
            s_logvar = torch.zeros(num_samples, mod.style_dim).to(self.flags.device)
            styles[m_key] = [s_mu, s_logvar]
        return styles

    def get_random_styles(self, num_samples):
        styles = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            if self.flags.factorized_representation:
                mod = self.modalities[m_key]
                z_style = torch.randn(num_samples, mod.style_dim)
                z_style = z_style.to(self.flags.device)
            else:
                z_style = None
            styles[m_key] = z_style
        return styles


from multivae.models.nn.default_architectures import (
    BaseAEConfig,
    Decoder_AE_MLP,
    Encoder_VAE_MLP,
    ModelOutput,
)

flags = ModelOutput(
    factorized_representation=False,
    device="cpu",
    batch_size=2,
    class_dim=10,
    modality_poe=False,
)


modalities = dict(
    m0=ModelOutput(
        encoder=Encoder_VAE_MLP(BaseAEConfig(input_dim=(2,), latent_dim=10)),
        decoder=Decoder_AE_MLP(BaseAEConfig(input_dim=(2,), latent_dim=10)),
        likelihood=torch.distributions.Laplace,
        name="m0",
    ),
    m1=ModelOutput(
        encoder=Encoder_VAE_MLP(BaseAEConfig(input_dim=(3,), latent_dim=10)),
        decoder=Decoder_AE_MLP(BaseAEConfig(input_dim=(3,), latent_dim=10)),
        likelihood=torch.distributions.Laplace,
        name="m1",
    ),
)

model = BaseMMVae(flags, modalities)

data = dict(
    m0=torch.tensor([[1, 2], [3, 4]]).float(),
    m1=torch.tensor([[1, 2, 5], [3, 4, 6]]).float(),
)

output = model(data)

print(output)

print(output["rec"]["m0"].log_prob(data["m0"]).sum(-1).mean())
print(output["rec"]["m1"].log_prob(data["m1"]).sum(-1).mean())


from multivae.models import MoPoE, MoPoEConfig

config = MoPoEConfig(
    n_modalities=2,
    latent_dim=10,
    decoders_dist=dict(m0="laplace", m1="laplace"),
    decoder_dist_params=dict(m0=dict(scale=0.75), m1=dict(scale=0.75)),
)

encoders = dict(m0=modalities["m0"].encoder, m1=modalities["m1"].encoder)

decoders = dict(m0=modalities["m0"].decoder, m1=modalities["m1"].decoder)


model = MoPoE(config, encoders, decoders)
from multivae.data.datasets import MultimodalBaseDataset

print(model(MultimodalBaseDataset(data=data)))


def calc_log_prob(out_dist, target, norm_value):
    log_prob = out_dist.log_prob(target).sum()
    mean_val_logprob = log_prob / norm_value
    return mean_val_logprob


def calc_log_probs(flags, modalities, result, batch):
    mods = modalities
    log_probs = dict()
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        log_probs[mod.name] = -calc_log_prob(
            result["rec"][mod.name], batch[mod.name], flags.batch_size
        )
        weighted_log_prob += log_probs[mod.name]
    return log_probs, weighted_log_prob


def calc_klds(flags, result):
    latents = result["latents"]["subsets"]
    klds = dict()
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar, norm_value=flags.batch_size)
    return klds


def basic_routine_epoch(flags, modalities, results, batch):
    log_probs, weighted_log_prob = calc_log_probs(flags, modalities, results, batch)
    group_divergence = results["joint_divergence"]

    klds = calc_klds(flags, results)

    kld_style = 0.0
    kld_content = group_divergence
    kld_weighted = 1 * kld_style + 1 * kld_content
    total_loss = 1 * weighted_log_prob + 1 * kld_weighted

    out_basic_routine = dict()
    out_basic_routine["weighted_log_probs"] = weighted_log_prob
    out_basic_routine["joint_divergence"] = kld_weighted

    out_basic_routine["total_loss"] = total_loss
    out_basic_routine["klds"] = klds
    return out_basic_routine


print(basic_routine_epoch(flags, modalities, output, data))
