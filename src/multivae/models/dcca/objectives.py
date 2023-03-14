from typing import List

import torch
from torch import Tensor


class cca_loss:
    def __init__(self, outdim_size, use_all_singular_values):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values

    def loss(self, H_list: List[Tensor]):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9
        H1, H2 = H_list[0], H_list[1]
        H1, H2 = H1.t(), H2.t()
        device = H1.device
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        #         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(
            o1
        ).to(device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(
            o2
        ).to(device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        # [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO="U")
        [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO="U")
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1**-0.5).to(device)), V1.t()
        )
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2**-0.5).to(device)), V2.t()
        )

        Tval = torch.matmul(
            torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv
        )
        #         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(
                trace_TT, (torch.eye(trace_TT.shape[0]).to(device) * r1)
            )  # regularization for more stability
            U = torch.linalg.eigvalsh(trace_TT, UPLO="U")
            U = torch.where(U > eps, U, (torch.ones(U.shape).double().to(device) * eps))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class mcca_loss:

    """
    Wraps-up several cca loss when training with more than two modalities.
    """

    def __init__(self, outdim_size, use_all_singular_values) -> None:
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.cca_loss = cca_loss(outdim_size, use_all_singular_values).loss

    def loss(self, H_list):
        loss = 0
        for i, h1 in enumerate(H_list):
            for j, h2 in enumerate(H_list):
                if i < j:
                    loss += self.cca_loss([h1, h2])

        return loss
