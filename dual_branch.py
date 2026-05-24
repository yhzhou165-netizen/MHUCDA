import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import Data

from basicModules import MLP, GCN, BnodeEmbedding, MultiHeadAttention


def dgl_to_pyg(g, h):
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0).to(g.device)
    return Data(x=h, edge_index=edge_index)


def pro_data(data, em, ed):
    edgeData = data.t()
    m_index = edgeData[0]
    d_index = edgeData[1]
    Em = torch.index_select(em, 0, m_index)
    Ed = torch.index_select(ed, 0, d_index)
    return Em, Ed


class HierarchicalSemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(HierarchicalSemanticAttention, self).__init__()
        self.intra_meta_attention = nn.MultiheadAttention(
            embed_dim=in_size, num_heads=4, dropout=dropout, batch_first=True
        )
        self.inter_meta_attention = nn.MultiheadAttention(
            embed_dim=in_size, num_heads=4, dropout=dropout, batch_first=True
        )
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, in_size)
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(in_size) for _ in range(num_layers + 2)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(in_size, in_size)

    def forward(self, z):
        intra_attended, intra_weights = self.intra_meta_attention(z, z, z)
        intra_enhanced = self.layer_norms[0](z + self.dropout(intra_attended))
        inter_attended, inter_weights = self.inter_meta_attention(
            intra_enhanced, intra_enhanced, intra_enhanced
        )
        inter_enhanced = self.layer_norms[1](
            intra_enhanced + self.dropout(inter_attended)
        )
        current = inter_enhanced
        for i, fusion_layer in enumerate(self.fusion_layers):
            residual = current
            avg_pool = current.mean(dim=1, keepdim=True).expand_as(current)
            max_pool, _ = current.max(dim=1, keepdim=True)
            max_pool = max_pool.expand_as(current)
            fused = torch.cat([current, avg_pool + max_pool], dim=-1)
            current = fusion_layer(fused)
            if i > 0:
                current = self.layer_norms[i + 2](residual + self.dropout(current))
        importance_scores = F.softmax(current.mean(dim=-1), dim=-1)
        final_output = (current * importance_scores.unsqueeze(-1)).sum(dim=1)
        final_output = self.output_proj(final_output)
        return final_output, (intra_weights, inter_weights, importance_scores)


class CooperativeAttention(nn.Module):
    def __init__(self, in_size, out_size, num_heads=4, dropout=0.1):
        super(CooperativeAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_size // num_heads
        assert self.head_dim * num_heads == out_size
        self.m2d_proj = nn.Linear(in_size, out_size)
        self.d2m_proj = nn.Linear(in_size, out_size)
        self.attention_m2d = nn.MultiheadAttention(
            embed_dim=out_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attention_d2m = nn.MultiheadAttention(
            embed_dim=out_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_m = nn.LayerNorm(out_size)
        self.layer_norm_d = nn.LayerNorm(out_size)
        self.gate_m = nn.Sequential(nn.Linear(out_size * 2, out_size), nn.Sigmoid())
        self.gate_d = nn.Sequential(nn.Linear(out_size * 2, out_size), nn.Sigmoid())

    def forward(self, h_m, h_d):
        h_m_proj = self.m2d_proj(h_m).unsqueeze(0)
        h_d_proj = self.d2m_proj(h_d).unsqueeze(0)
        h_d_attended, weights_m2d = self.attention_m2d(h_d_proj, h_m_proj, h_m_proj)
        h_m_attended, weights_d2m = self.attention_d2m(h_m_proj, h_d_proj, h_d_proj)
        h_m_enhanced = self.layer_norm_m(
            h_m_proj.squeeze(0) + self.gate_m(
                torch.cat([h_m_proj.squeeze(0), h_m_attended.squeeze(0)], dim=-1)
            ) * h_m_attended.squeeze(0)
        )
        h_d_enhanced = self.layer_norm_d(
            h_d_proj.squeeze(0) + self.gate_d(
                torch.cat([h_d_proj.squeeze(0), h_d_attended.squeeze(0)], dim=-1)
            ) * h_d_attended.squeeze(0)
        )
        return h_m_enhanced, h_d_enhanced, (weights_m2d, weights_d2m)


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, dropout):
        super().__init__()
        self.meta_paths = list(tuple(mp) for mp in meta_paths)
        self.gin_layers = nn.ModuleList()
        for _ in meta_paths:
            mlp = nn.Sequential(
                nn.Linear(in_size, in_size),
                nn.BatchNorm1d(in_size),
                nn.ReLU(),
                nn.Linear(in_size, out_size * 2),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.semantic_attention = HierarchicalSemanticAttention(
            in_size=out_size * 2, hidden_size=64, num_layers=2, dropout=dropout
        )

    def forward(self, g, h):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp in self.meta_paths:
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, mp)
        device = h.device
        semantic_embeddings = []
        for i, mp in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[mp].to(device)
            pyg_data = dgl_to_pyg(new_g, h)
            out = self.gin_layers[i](pyg_data.x, pyg_data.edge_index)
            semantic_embeddings.append(out)
        stacked = torch.stack(semantic_embeddings, dim=1)
        final_output, attn_weights = self.semantic_attention(stacked)
        self.attention_weights = attn_weights
        return final_output


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN, self).__init__()
        self.layer = HANLayer(meta_paths, in_size, hidden_size, dropout)
        self.predict = nn.Linear(hidden_size * 2, out_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, g, h):
        h = self.layer(g, h)
        return self.predict(h)


class CollaborativeHAN_MDA(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(CollaborativeHAN_MDA, self).__init__()
        self.shared_miRNA_han = HAN(
            all_meta_paths[0], in_size, hidden_size, hidden_size, dropout
        )
        self.shared_disease_han = HAN(
            all_meta_paths[1], in_size, hidden_size, hidden_size, dropout
        )
        self.cooperative_attention = CooperativeAttention(
            hidden_size, hidden_size, num_heads=4, dropout=dropout
        )
        self.miRNA_specific = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, out_size)
        )
        self.disease_specific = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, out_size)
        )
        self.contrastive_loss_weight = 0.1
        self.temperature = 0.5
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, s_g, s_h_1, s_h_2, positive_pairs=None, return_contrastive=False):
        shared_miRNA = self.shared_miRNA_han(s_g[0], s_h_1)
        shared_disease = self.shared_disease_han(s_g[1], s_h_2)
        coattended_m, coattended_d, coattention_weights = \
            self.cooperative_attention(shared_miRNA, shared_disease)
        h_miRNA = self.miRNA_specific(
            torch.cat([shared_miRNA, coattended_m], dim=-1)
        )
        h_disease = self.disease_specific(
            torch.cat([shared_disease, coattended_d], dim=-1)
        )
        contrastive_loss = 0.0
        if return_contrastive and positive_pairs is not None:
            contrastive_loss = self._contrastive_loss(h_miRNA, h_disease, positive_pairs)
        if return_contrastive:
            return h_miRNA, h_disease, coattention_weights, contrastive_loss
        else:
            return h_miRNA, h_disease, coattention_weights

    def _contrastive_loss(self, h_miRNA, h_disease, positive_pairs):
        pos_loss = 0
        batch_size = len(positive_pairs)
        if batch_size == 0:
            return torch.tensor(0.0, device=h_miRNA.device)
        for (m_idx, d_idx) in positive_pairs:
            if m_idx < h_miRNA.shape[0] and d_idx < h_disease.shape[0]:
                m_embed = h_miRNA[m_idx]
                d_embed = h_disease[d_idx]
                pos_sim = F.cosine_similarity(
                    m_embed.unsqueeze(0), d_embed.unsqueeze(0), dim=1
                )
                pair_loss = torch.clamp(
                    -torch.log(torch.sigmoid(pos_sim / self.temperature)), max=3.0
                )
                pos_loss += pair_loss
        return pos_loss / batch_size


class MD_hyper(nn.Module):
    def __init__(self, param):
        super(MD_hyper, self).__init__()
        self.inSize = param.inSize
        self.outSize = param.outSize
        self.hiddenSize = param.hiddenSize
        self.gcnlayers = param.gcn_layers
        self.device = param.device
        self.PVN = param.PVN
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.maskMDA = param.maskMDA
        self.realnode = param.batchSize
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()
        self.num_heads1 = param.num_heads1
        self.nodeEmbedding = BnodeEmbedding(
            torch.tensor(
                np.random.normal(
                    size=(max(int(self.PVN * self.realnode), 0), self.inSize)
                ),
                dtype=torch.float32
            ),
            dropout=self.hdnDropout
        ).to(self.device)
        self.nodeGCN = GCN(
            self.inSize, self.outSize,
            dropout=self.hdnDropout, layers=self.gcnlayers,
            resnet=True, actFunc=self.relu1
        ).to(self.device)
        self.fcLinear = MLP(
            self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1
        ).to(self.device)
        self.layeratt_m = MultiHeadAttention(
            self.inSize, self.outSize, self.gcnlayers, self.num_heads1
        )
        self.layeratt_d = MultiHeadAttention(
            self.inSize, self.outSize, self.gcnlayers, self.num_heads1
        )

    def forward(self, em, ed):
        xm = em.unsqueeze(1)
        xd = ed.unsqueeze(1)
        if self.PVN > 0:
            node = self.nodeEmbedding.dropout2(
                self.nodeEmbedding.dropout1(
                    self.nodeEmbedding.embedding.weight
                )
            ).repeat(len(xd), 1, 1)
            node = torch.cat([xm, xd, node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (
                nodeDist * nodeDist.transpose(1, 2) + 1e-8
            )
            cosNode = self.relu2(cosNode)
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskMDA:
                cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(
                node.shape[1], dtype=torch.float32, device=self.device
            ).repeat(len(xm), 1, 1)
            D[:, range(node.shape[1]), range(node.shape[1])] = \
                1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)
            mGCNem, dGCNem = self.nodeGCN(node, pL)
            mLAem = self.layeratt_m(mGCNem)
            dLAem = self.layeratt_d(dGCNem)
            node_embed = (mLAem * dLAem).squeeze(dim=1)
        else:
            node_embed = (xm * xd).squeeze(dim=1)
        pre_part = self.fcLinear(node_embed)
        pre_a = self.sigmoid(pre_part).squeeze(dim=1)
        return pre_a


class HHN(nn.Module):
    def __init__(self, param):
        super(HHN, self).__init__()
        self.inSize = param.inSize
        self.outSize = param.outSize
        self.hiddenSize = param.hiddenSize
        self.device = param.device
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.md_hyper = MD_hyper(param)
        self.num_heads1 = param.num_heads1
        self.fcLinear = MLP(
            self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1
        ).to(self.device)
        self.gcn = GCNConv(128, 128)

    def forward(self, sdata, tdata, em, ed):
        x = torch.cat([em, ed], dim=0)
        edge_index = sdata['m_d']['edges']
        edge_weight = sdata['m_d']['data_matrix'][
            sdata['m_d']['edges'][0],
            sdata['m_d']['edges'][1]
        ]
        if x.device != edge_index.device:
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)
        out = torch.relu(self.gcn(x, edge_index, edge_weight))
        out1 = torch.relu(self.gcn(out, edge_index, edge_weight))
        out_m = out1[:561, :]
        out_d = out1[561:, :]
        mFea, dFea = pro_data(tdata, out_m, out_d)
        pre = self.md_hyper(mFea, dFea)
        return pre