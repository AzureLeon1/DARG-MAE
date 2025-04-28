import copy

import torch
import numpy
import torch
import torch.nn as nn
from typing import Optional
from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
import torch.nn.functional as F
from dargmae.utils import create_norm, drop_edge

from torch_geometric.nn import global_add_pool

import dgl


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


def get_sub_prob(gnn_emb, subgraph_id, fc):


    unique_subgraph, new_subgraph_id = torch.unique(subgraph_id, return_inverse=True)
    subgraph_emb = global_add_pool(gnn_emb, new_subgraph_id)
    subgraph_logits = fc(subgraph_emb)  # (num_subgraphs, 2)
    subgraph_prob = F.gumbel_softmax(subgraph_logits, hard=True)  # (num_subgraphs, 2)
    node_prob = subgraph_prob[new_subgraph_id]

    return node_prob

class sub_AdversMask(nn.Module):
    def __init__(self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            sub_mask_encoder_type: str = "gcn",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            max_degree: int = 170,
            emb_dim: int = 16):
        super(sub_AdversMask, self).__init__()
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        self.sub_mask_encoder_type = sub_mask_encoder_type
        if sub_mask_encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        num_class = 2

        self.mask = setup_module(
            m_type=sub_mask_encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.fc = nn.Linear(enc_num_hidden*nhead , num_class)
        self.fc_mlp = nn.Linear(enc_num_hidden, num_class)

    def forward(self, graph, x,args):
        subgraph_id = graph.ndata['subgraph_id']
        if args.mask_encoder == 'gat':
            gnn_emb = self.mask(graph,x)
            z = get_sub_prob(gnn_emb, subgraph_id, self.fc)

        elif args.mask_encoder == 'mlp':
            mlp_emb = self.mask(x)
            z = get_sub_prob(mlp_emb, subgraph_id, self.fc_mlp)

        else:
            gnn_emb = self.mask(graph,x)
            z = get_sub_prob(gnn_emb, subgraph_id, self.fc_mlp)

        return z

