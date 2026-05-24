import torch
import torch.nn as nn

from basicModules import MLP, MultiHeadAttention
from hypergraph_encoder import (ContrastiveMultiViewEmbeddingM,
                                ContrastiveMultiViewEmbeddingD)
from dual_branch import CollaborativeHAN_MDA, HHN, pro_data
from adaptive_fusion import AdaptiveGatedFusion


class MHUCDA(nn.Module):
    def __init__(self, param, m_emd, d_emd, graph_data, all_meta_paths):
        super(MHUCDA, self).__init__()
        self.inSize = param.inSize
        self.hiddenSize = param.hiddenSize
        self.outSize = param.outSize
        self.Dropout = param.Dropout
        self.device = param.device
        self.graph = graph_data

        self.Xm = m_emd
        self.Xd = d_emd

        self.xm_embed = nn.Parameter(torch.empty(561, param.fm))
        self.xd_embed = nn.Parameter(torch.empty(100, param.fd))
        nn.init.normal_(self.xm_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.xd_embed, mean=0.0, std=0.02)

        self.MHM = CollaborativeHAN_MDA(
            all_meta_paths, self.inSize,
            self.hiddenSize, self.outSize, self.Dropout
        ).to(self.device)

        self.HHN = HHN(param)

        self.layeratt_m = MultiHeadAttention(
            param.inSize, param.outSize, 2, param.num_heads1
        )
        self.layeratt_d = MultiHeadAttention(
            param.inSize, param.outSize, 2, param.num_heads1
        )

        self.fcLinear = MLP(
            self.outSize, 1, dropout=0.1, actFunc=nn.ReLU()
        ).to(self.device)
        self.sigmoid = nn.Sigmoid()

        self.adaptive_fusion = AdaptiveGatedFusion(dim=param.outSize)

        self.use_contrastive = True
        self.contrastive_weight = 0.25

    def forward(self, sim_data, train_data,
                return_contrastive=False, train_labels=None):
        xm = self.xm_embed.to(self.device)
        xd = self.xd_embed.to(self.device)

        Em, cross_view_loss_m = self.Xm(sim_data, xm)
        Ed, cross_view_loss_d = self.Xd(sim_data, xd)
        cross_view_loss = (cross_view_loss_m + cross_view_loss_d) * 0.05

        positive_pairs = None
        if self.use_contrastive and return_contrastive and train_labels is not None:
            positive_indices = torch.where(train_labels == 1)[0]
            valid_indices = [
                i for i in positive_indices if i < train_data.shape[0]
            ]
            positive_pairs = [
                (int(train_data[i, 0]), int(train_data[i, 1]))
                for i in valid_indices
            ]

        if return_contrastive and positive_pairs:
            h_11, h_21, attention_weights, han_cl_loss = self.MHM(
                self.graph, Em, Ed,
                positive_pairs=positive_pairs,
                return_contrastive=True
            )
            contrastive_loss = han_cl_loss + cross_view_loss
        else:
            h_11, h_21, attention_weights = self.MHM(self.graph, Em, Ed)
            contrastive_loss = cross_view_loss

        mFea, dFea = pro_data(train_data, h_11, h_21)
        out_m = self.layeratt_m(mFea)
        out_d = self.layeratt_d(dFea)
        node_embed = out_m * out_d
        pre_part = self.fcLinear(node_embed)
        pre_MHM = self.sigmoid(pre_part).squeeze(dim=1)

        pre_HHN = self.HHN(sim_data, train_data, Em, Ed)

        edge_data = train_data.t()
        h_m = Em[edge_data[0]]
        h_d = Ed[edge_data[1]]

        pre_final, uncertainty = self.adaptive_fusion(h_m, h_d, pre_MHM, pre_HHN)

        self.attention_weights = attention_weights
        self.last_uncertainty = uncertainty.detach()

        if return_contrastive:
            return pre_final, contrastive_loss
        else:
            return pre_final

    def get_attention_weights(self):
        return getattr(self, 'attention_weights', None)

    def get_uncertainty(self):
        return getattr(self, 'last_uncertainty', None)