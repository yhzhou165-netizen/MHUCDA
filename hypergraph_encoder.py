import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 10):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self._theta_cache: dict = {}

    def _build_theta(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        n = sim_matrix.shape[0]
        k = min(self.k, n - 1)
        device = sim_matrix.device
        _, topk_idx = torch.topk(sim_matrix, k + 1, dim=1)
        H = torch.zeros(n, n, device=device)
        node_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, k + 1)
        H[topk_idx.reshape(-1), node_idx.reshape(-1)] = 1.0
        H[torch.arange(n, device=device), torch.arange(n, device=device)] = 1.0
        dv = H.sum(dim=1).clamp(min=1e-6)
        de = H.sum(dim=0).clamp(min=1e-6)
        dv_inv_sqrt = dv.pow(-0.5)
        de_inv = de.pow(-1.0)
        A = dv_inv_sqrt.unsqueeze(1) * H
        B = (de_inv.unsqueeze(0) * H).t()
        theta = (A @ B) * dv_inv_sqrt.unsqueeze(0)
        return theta

    def forward(self, x: torch.Tensor, sim_matrix: torch.Tensor) -> torch.Tensor:
        mat_id = id(sim_matrix)
        if mat_id not in self._theta_cache:
            with torch.no_grad():
                self._theta_cache[mat_id] = self._build_theta(sim_matrix.detach())
        theta = self._theta_cache[mat_id].to(x.device)
        agg = theta @ x
        out = self.linear(agg)
        out = self.bn(out)
        return F.relu(out)


class ContrastiveMultiViewEmbeddingM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = getattr(args, 'cl_temperature', 0.5)
        k = getattr(args, 'k_neighbors', 10)
        self.encoders = nn.ModuleDict({
            'functional': HypergraphConvLayer(args.fm, args.fm, k=k),
            'sequence':   HypergraphConvLayer(args.fm, args.fm, k=k),
            'gaussian':   HypergraphConvLayer(args.fm, args.fm, k=k),
        })
        self.projector = nn.Sequential(
            nn.Linear(args.fm, args.fm),
            nn.ReLU(),
            nn.Linear(args.fm, args.fm // 2),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(args.fm * 3, args.fm),
            nn.ReLU(),
            nn.Linear(args.fm, args.fm),
        )

    def _cross_view_infoNCE(self, view_embs: list) -> torch.Tensor:
        proj = [F.normalize(self.projector(e), dim=1) for e in view_embs]
        loss = torch.tensor(0.0, device=proj[0].device)
        count = 0
        for i in range(len(proj)):
            for j in range(i + 1, len(proj)):
                zi = proj[i]
                zj = proj[j]
                logits = torch.mm(zi, zj.t()) / self.temperature
                labels = torch.arange(zi.shape[0], device=zi.device)
                loss_ij = F.cross_entropy(logits, labels)
                loss_ji = F.cross_entropy(logits.t(), labels)
                loss = loss + (loss_ij + loss_ji) * 0.5
                count += 1
        return loss / max(count, 1)

    def forward(self, data, fm1):
        views = {
            'functional': data['mm_f']['data_matrix'],
            'sequence':   data['mm_s']['data_matrix'],
            'gaussian':   data['mm_g']['data_matrix'],
        }
        view_embs = {}
        for name, sim_matrix in views.items():
            sim = sim_matrix.to(fm1.device)
            view_embs[name] = self.encoders[name](fm1, sim)
        cl_loss = torch.tensor(0.0, device=fm1.device)
        if torch.is_grad_enabled():
            cl_loss = self._cross_view_infoNCE(list(view_embs.values()))
        concatenated = torch.cat(list(view_embs.values()), dim=1)
        fused = self.fusion_net(concatenated)
        return fused, cl_loss


class ContrastiveMultiViewEmbeddingD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = getattr(args, 'cl_temperature', 0.5)
        k = getattr(args, 'k_neighbors', 10)
        self.encoders = nn.ModuleDict({
            'temporal': HypergraphConvLayer(args.fd, args.fd, k=k),
            'spatial':  HypergraphConvLayer(args.fd, args.fd, k=k),
            'gaussian': HypergraphConvLayer(args.fd, args.fd, k=k),
        })
        self.projector = nn.Sequential(
            nn.Linear(args.fd, args.fd),
            nn.ReLU(),
            nn.Linear(args.fd, args.fd // 2),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(args.fd * 3, args.fd),
            nn.ReLU(),
            nn.Linear(args.fd, args.fd),
        )

    def _cross_view_infoNCE(self, view_embs: list) -> torch.Tensor:
        proj = [F.normalize(self.projector(e), dim=1) for e in view_embs]
        loss, count = torch.tensor(0.0, device=proj[0].device), 0
        for i in range(len(proj)):
            for j in range(i + 1, len(proj)):
                zi, zj = proj[i], proj[j]
                logits = torch.mm(zi, zj.t()) / self.temperature
                labels = torch.arange(zi.shape[0], device=zi.device)
                loss += (F.cross_entropy(logits, labels) +
                         F.cross_entropy(logits.t(), labels)) * 0.5
                count += 1
        return loss / max(count, 1)

    def forward(self, data, dm1):
        views = {
            'temporal': data['dd_t']['data_matrix'],
            'spatial':  data['dd_s']['data_matrix'],
            'gaussian': data['dd_g']['data_matrix'],
        }
        view_embs = {}
        for name, sim_matrix in views.items():
            sim = sim_matrix.to(dm1.device)
            view_embs[name] = self.encoders[name](dm1, sim)
        cl_loss = torch.tensor(0.0, device=dm1.device)
        if torch.is_grad_enabled():
            cl_loss = self._cross_view_infoNCE(list(view_embs.values()))
        concatenated = torch.cat(list(view_embs.values()), dim=1)
        fused = self.fusion_net(concatenated)
        return fused, cl_loss