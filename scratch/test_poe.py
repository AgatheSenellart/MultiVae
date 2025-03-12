import torch

mus = [torch.randn(128,64)*10, torch.randn(128,64)*10]
lvs = [torch.randn(128,64)*10, torch.randn(128,64)*10]

def poe1( mu, logvar, eps=1e-9):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1.0 / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

def poe2(mus_list, log_vars_list):
        mus = mus_list.copy()
        log_vars = log_vars_list.copy()

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars])  # Compute the inverse of variances
        lnV = -torch.logsumexp(lnT, dim=0)  # variances of the product of expert
        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        return joint_mu, lnV


print(torch.max(torch.abs((poe1(torch.stack(mus), torch.stack(lvs))[0]- poe2(mus, lvs)[0]))))
print(torch.max(torch.abs((poe1(torch.stack(mus), torch.stack(lvs))[1]- poe2(mus, lvs)[1]))))