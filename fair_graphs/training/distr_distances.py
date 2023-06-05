import torch as tr
from torch import nn


# -----------------------------------
# --- Fairness Distribution Distances
# -----------------------------------

class FirstOrderMatching(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", 'sum', "norm"):
            raise ValueError("Bad argument value for the reduction parameter")
        self.reduction = reduction

    def __str__(self):
        return "FOM"

    @staticmethod
    def mean_distance(d1, d2, reduction="mean"):
        loss = (d1.mean(dim=0) - d2.mean(dim=0)).abs()
        
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "norm":
            loss = tr.linalg.norm(loss, ord=2)
        else:
            loss = loss.mean()
        
        return loss

    def forward(self, distr_1, distr_2):
        return FirstOrderMatching.mean_distance(distr_1, distr_2, reduction=self.reduction)


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def __str__(self):
        return "SD"
    
    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = tr.sum((tr.abs(x_col - y_lin)) ** p, -1)
        return C

    #@staticmethod
    #def ave(u, u1, tau):
    #    "Barycenter subroutine, used by kinetic acceleration through extrapolation."
    #    return tau * u + (1 - tau) * u1
    
    @staticmethod
    def sinkhorn_distance(x, y, C, max_iter, eps, reduction="mean"):
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        device = x.device
        thresh = 1e-1   # HARD-CODED stopping criterion

        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
            
        # both marginals are fixed with equal weights
        mu = tr.empty(batch_size, x_points, dtype=tr.float,
                      requires_grad=False, device=device).fill_(1.0/x_points).squeeze()
        nu = tr.empty(batch_size, y_points, dtype=tr.float,
                      requires_grad=False, device=device).fill_(1.0/y_points).squeeze()
        u = tr.zeros_like(mu)
        v = tr.zeros_like(nu)
        
        def m_cost(C, u, v, eps):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps
        
        for i in range(max_iter): # Sinkhorn iterations
            last_u = u
            u = eps * (tr.log(mu+1e-8) - tr.logsumexp(m_cost(C, u, v, eps), dim=-1)) + u
            v = eps * (tr.log(nu+1e-8) - tr.logsumexp(m_cost(C, u, v, eps).transpose(-2, -1), dim=-1)) + v
            err = (u - last_u).abs().sum(-1).mean()
            if err.item() < thresh:
                break
            
        U, V = u, v
        pi = tr.exp(m_cost(C, U, V, eps)) # Transport plan pi = diag(a)*K*diag(b)
        cost = tr.sum(pi*C, dim=(-2, -1)) # Sinkhorn distance

        if reduction == 'mean':
            cost = cost.mean()
        elif reduction == 'sum':
            cost = cost.sum()

        return cost#, pi
        
    
    def forward(self, distr_1, distr_2):
        cost_mtx = SinkhornDistance._cost_matrix(distr_1, distr_2)  # Wasserstein cost function
        return SinkhornDistance.sinkhorn_distance(distr_1, distr_2,
                                                  C = cost_mtx,
                                                  max_iter = self.max_iter,
                                                  eps = self.eps,
                                                  reduction = self.reduction)


class GaussMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def __str__(self):
        return "MMD"
    
    @staticmethod
    def gauss_mmd(d1, d2, gamma):
        n1, n2 = d1.shape[0], d2.shape[0]
        d = d1.shape[1]
        assert d == d2.shape[1], "the two distributions don't have the same dimensionality"
        
        pdist_d1 = tr.linalg.norm(d1.unsqueeze(-1)-d1, dim=2, ord=2)**2
        pdist_d2 = tr.linalg.norm(d2.unsqueeze(-1)-d2, dim=2, ord=2)**2
        pdist_d1d2 = tr.linalg.norm(d1.unsqueeze(-1)-d2, dim=2, ord=2)**2
        
        gauss_krn1 = (-gamma*(pdist_d1)/d).exp().sum(dim=(0,1)) / (n1**2)
        gauss_krn2 = (-gamma*(pdist_d2)/d).exp().sum(dim=(0,1)) / (n2**2)
        gauss_sim = 2*(-gamma*(pdist_d1d2)/d).exp().sum(dim=(0,1)) / (n1*n2)
                       
        dist = gauss_krn1 + gauss_krn2 - gauss_sim
        return dist
    
    def forward(self, distribution1, distribution2):
        return GaussMaximumMeanDiscrepancy.gauss_mmd(distribution1, distribution2, gamma=self.gamma)
