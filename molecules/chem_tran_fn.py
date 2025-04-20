from torch_geometric.data import InMemoryDataset
import torch_geometric
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
import os
from itertools import repeat, product, chain
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
import copy
import math
import logging


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def neg_sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def train_mae(args, model_list, loader, optimizer_list, device, epoch, alpha_l=1.0, loss_fn="sce"):
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
        mask_criterion = partial(neg_sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()
        mask_criterion = nn.CrossEntropyLoss()

    model, adv_mask, adv_sub_mask, dec_pred_atoms, dec_pred_bonds = model_list
    optimizer_model, optimizer_adv_mask, optimizer_adv_sub_mask, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds = optimizer_list

    model.train()
    adv_mask.train()
    adv_sub_mask.train()
    dec_pred_atoms.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    alpha_adv = args.alpha_0 + \
        (((epoch-1) / args.epochs) ** args.gamma) * (args.alpha_T - args.alpha_0)
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        batch_tmp = copy.deepcopy(batch)

        # mask_prob = adv_mask(batch.x, batch.edge_index, batch.edge_attr)
        node_mask_prob = adv_mask(batch)
        sub_mask_prob = adv_sub_mask(batch)
        mask_prob = args.sub_weight * sub_mask_prob + \
            (1 - args.sub_weight) * node_mask_prob
        node_rep, u_loss = model(batch, mask_prob, alpha_adv, args)

        # loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        pred_node = dec_pred_atoms(
            node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)

        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if loss_fn == "sce":
            # loss = criterion(node_attr_label, pred_node[masked_node_indices])
            loss_mask = mask_criterion(
                node_attr_label, pred_node[masked_node_indices])
        else:
            loss_mask = criterion(
                pred_node.double()[masked_node_indices], batch.mask_node_label[:, 0])

        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        # acc_node_accum += acc_node

        if args.mask_edge:

            edge_rep = node_rep[batch.edge_index[0]] + \
                node_rep[batch.edge_index[1]]
            pred_edge = dec_pred_bonds(
                edge_rep, batch.edge_index, batch.edge_attr, batch.connected_edge_indices)
            loss_mask += mask_criterion(
                pred_edge[batch.connected_edge_indices].double(), batch.edge_attr_label)

        # loss_mask = -loss_mask + args.belta*(torch.tensor([1.]).to(device)/torch.sin(torch.pi/ len(batch.x) * (mask_prob[:,0].sum())))
        loss_mask = -loss_mask + args.belta * \
            (torch.tensor([1.]).to(device) / torch.sin(math.pi /
             len(batch.x) * (mask_prob[:, 0].sum())))

        optimizer_model.zero_grad()
        optimizer_adv_mask.zero_grad()
        optimizer_adv_sub_mask.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss_mask.backward()

        optimizer_adv_mask.step()
        optimizer_adv_sub_mask.step()
        batch = batch_tmp

        optimizer_model.zero_grad()
        optimizer_adv_mask.zero_grad()
        optimizer_adv_sub_mask.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        node_mask_prob = adv_mask(batch)
        sub_mask_prob = adv_sub_mask(batch)
        mask_prob = args.sub_weight * sub_mask_prob + \
            (1 - args.sub_weight) * node_mask_prob
        node_rep, u_loss = model(batch, mask_prob, alpha_adv, args)

        # loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        pred_node = dec_pred_atoms(
            node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)

        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
            # loss_mask = mask_criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[
                             masked_node_indices], batch.mask_node_label[:, 0])

        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        # acc_node_accum += acc_node

        if args.mask_edge:
            edge_rep = node_rep[batch.edge_index[0]] + \
                node_rep[batch.edge_index[1]]
            pred_edge = dec_pred_bonds(
                edge_rep, batch.edge_index, batch.edge_attr, batch.connected_edge_indices)
            loss += criterion(
                pred_edge[batch.connected_edge_indices].double(), batch.edge_attr_label)
            # masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            # edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            # pred_edge = dec_pred_bonds(edge_rep)
            # loss += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])
            # # loss_mask += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])

        loss = loss + u_loss
        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum / step  # , acc_node_accum/step, acc_edge_accum/step


num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super().__init__()
        self._dec_type = gnn_type
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr="add")
        # elif gnn_type == "gcn":
        #     self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = torch.nn.PReLU()
        self.temp = 0.2

    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)

            x[mask_node_indices] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index, edge_attr)
            # out = F.softmax(out, dim=-1) / self.temp
        return out


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(
            edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, uniformity_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.x_lin1 = torch.nn.Linear(1, emb_dim)
        self.x_lin2 = torch.nn.Linear(1, emb_dim)
        self.uniformity_dim = uniformity_dim
        if self.JK == "concat":
            self.uniformity_layer = nn.Linear(
                emb_dim * self.num_layer, self.uniformity_dim, bias=False)
        else:
            self.uniformity_layer = nn.Linear(
                emb_dim, self.uniformity_dim, bias=False)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    # def forward(self, *argv):
    def forward(self, batch, mask_prob, alpha_adv, args):
        # batch, mask_prob, alpha_adv, args = argv[0], argv[1], argv[2], argv[3]
        # input(batch)
        # if len(argv) == 3:
        #     batch, mask_prob, alpha_adv, args = argv[0], argv[1], argv[2], argv[3]
        # elif len(argv) == 1:
        #     data = argv[0]
        #     x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # else:
        #     raise ValueError("unmatched number of arguments.")

        x = batch.x

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # print(batch)
        # print("pre")

        num_nodes = len(x)
        mask_num_nodes = len(batch.masked_atom_indices)
        num_random_mask_nodes = int(mask_num_nodes * (1. - alpha_adv))

        random_mask_nodes = batch.masked_atom_indices[:num_random_mask_nodes]
        random_keep_nodes = batch.masked_atom_indices[num_random_mask_nodes:]

        mask_ = mask_prob[:, 1]
        perm_adv = torch.randperm(num_nodes, device=x.device)
        adv_keep_token = perm_adv[:int(num_nodes * (1. - alpha_adv))]
        mask_[adv_keep_token] = 1.
        Mask_ = mask_.reshape(-1, 1)
        adv_keep_nodes = mask_.nonzero().reshape(-1)
        adv_mask_nodes = (1 - mask_).nonzero().reshape(-1)

        mask_nodes = torch.cat(
            (random_mask_nodes, adv_mask_nodes), dim=0).unique()
        keep_nodes = torch.tensor(np.intersect1d(
            random_keep_nodes.cpu().numpy(), adv_keep_nodes.cpu().numpy())).to(x.device)
        num_mask_nodes = mask_nodes.shape[0]

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        if args.replace_rate > 0:
            num_noise_nodes = int(args.replace_rate * num_mask_nodes)

            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(
                (1 - args.replace_rate) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(
                args.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = self.x_embedding1(torch.tensor(119).to(
                x.device)) + self.x_embedding2(torch.tensor(0).to(x.device))
            one = torch.ones_like(x, dtype=torch.float)
            # one = torch.ones_like(x, dtype = torch.long)
            x = x * one
            out_x = out_x * Mask_
            # out_x[token_nodes] = torch.tensor([119.0, 0]).to(x.device)
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            out_x = out_x * Mask_
            token_nodes = mask_nodes
            # out_x[token_nodes] = torch.tensor([119.0, 0]).to(x.device)
            out_x[token_nodes] = self.x_embedding1(torch.tensor(119).to(
                x.device)) + self.x_embedding2(torch.tensor(0).to(x.device))
            # out_x[token_nodes] = torch.tensor([119, 0]).to(x.device)

        x = out_x

        batch.mask_node_label = batch.x[mask_nodes]
        atom_type = F.one_hot(
            batch.mask_node_label[:, 0], num_classes=119).float()
        batch.node_attr_label = atom_type
        batch.x = x

        batch.masked_atom_indices = mask_nodes

        mask_nodes_tmp = copy.deepcopy(mask_nodes).to(torch.device("cpu"))

        if args.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms

            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(edge_index.cpu().numpy().T):
                for atom_idx in mask_nodes_tmp:
                    if atom_idx.item() in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            print(len(connected_edge_indices))
            connected_edge_indices = connected_edge_indices
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        edge_attr[bond_idx].view(1, -1))

                batch.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)

                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    batch.edge_attr[bond_idx] = torch.tensor([5, 0])

                batch.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                batch.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                batch.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(
                batch.mask_edge_label[:, 0], num_classes=5).float()
            bond_direction = F.one_hot(
                batch.mask_edge_label[:, 1], num_classes=3).float()
            batch.edge_attr_label = torch.cat(
                (edge_type, bond_direction), dim=1)

        if args.drop_edge_rate > 0:
            pass

        edge_attr = batch.edge_attr
        # x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1]) ##########
        # x = self.x_lin1(x[:, 0].reshape(-1,1)) + self.x_lin2(x[:, 1].reshape(-1,1))
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio,
                              training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        if keep_nodes.numel() == 0:
            u_loss = 0
        else:
            pooling = args.pooling
            if pooling == "mean":
                graph_emb = torch_geometric.nn.global_mean_pool(
                    node_representation[keep_nodes], batch.batch[keep_nodes])
            elif pooling == "max":
                graph_emb = torch_geometric.nn.global_max_pool(
                    node_representation[keep_nodes], batch.batch[keep_nodes])
            elif pooling == "sum":
                graph_emb = torch_geometric.nn.global_add_pool(
                    node_representation[keep_nodes], batch.batch[keep_nodes])
            else:
                raise NotImplementedError
            lamda = args.lamda * (1.0 - alpha_adv)
            # node_eb = F.relu(self.uniformity_layer(node_representation))
            node_eb = F.relu(self.uniformity_layer(graph_emb))
            u_loss = uniformity_loss(node_eb, lamda)

        return node_representation, u_loss


def uniformity_loss(node_rep, t, max_size=30000, batch=10000):
    # calculate loss
    n = node_rep.size(0)
    node_rep = torch.nn.functional.normalize(node_rep)
    if n < max_size:
        loss = torch.log(
            torch.exp(2. * t * ((node_rep @ node_rep.T) - 1.)).mean())
    else:
        total_loss = 0.
        permutation = torch.randperm(n)
        node_rep = node_rep[permutation]
        for i in range(0, n, batch):
            batch_features = node_rep[i:i + batch]
            batch_loss = torch.log(
                torch.exp(2. * t * ((batch_features @ batch_features.T) - 1.)).mean())
            total_loss += batch_loss
        loss = total_loss / (n // batch)
    return loss


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                    'possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


global_subgraph_counter = 0


def analyze_smiles_with_complete_indices(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    fragments = BRICS.BRICSDecompose(mol)

    fragment_atom_indices = []
    for fragment in fragments:
        print(fragment)
        frag_mol = Chem.MolFromSmiles(fragment)
        if frag_mol:
            frag_indices = []
            for atom in frag_mol.GetAtoms():
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num > 0:
                    frag_indices.append(atom_map_num - 1)
            fragment_atom_indices.append(frag_indices)

    atom_count = mol.GetNumAtoms()
    atom_to_subgraph = [-1] * atom_count
    subgraph_info = {}
    for subgraph_id, frag_atom_indices in enumerate(fragment_atom_indices):
        subgraph_info[subgraph_id] = frag_atom_indices
        for idx in frag_atom_indices:
            atom_to_subgraph[idx] = subgraph_id

    print(atom_to_subgraph)

    unique_subgraphs = sorted(set(atom_to_subgraph) - {-1})
    subgraph_mapping = {original: new for new,
                        original in enumerate(unique_subgraphs)}

    for i, subgraph_id in enumerate(atom_to_subgraph):
        if subgraph_id in subgraph_mapping:
            atom_to_subgraph[i] = subgraph_mapping[subgraph_id] + 1
        else:
            atom_to_subgraph[i] = -1

    # print(f"atom_to_subgraph: {atom_to_subgraph}")
    return atom_to_subgraph


def mol_to_graph_data_obj_with_brics(smiles):
    global global_subgraph_counter

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")

    atom_subgraph = analyze_smiles_with_complete_indices(smiles)

    data = mol_to_graph_data_obj_simple(mol)

    edge_index, edge_attr = data.edge_index, data.edge_attr

    edge_mask = []
    for i, (src, dst) in enumerate(edge_index.T):
        if not (atom_subgraph[src] == -1 and atom_subgraph[dst] == -1):
            edge_mask.append(i)

    edge_index2 = edge_index[:, edge_mask]
    edge_attr2 = edge_attr[edge_mask]

    max_index = max(atom_subgraph) if atom_subgraph else -1
    for idx, subgraph in enumerate(atom_subgraph):
        if subgraph == -1:
            neighbors = edge_index2[1][edge_index2[0] == idx]
            if len(neighbors) > 0:
                atom_subgraph[idx] = atom_subgraph[neighbors[0].item()]
            else:
                max_index = max_index + 1
                atom_subgraph[idx] = max_index
    # print(atom_subgraph)

    atom_subgraph = [subgraph +
                     global_subgraph_counter for subgraph in atom_subgraph]

    global_subgraph_counter += len(set(atom_subgraph))

    data.edge_index2 = edge_index2
    data.edge_attr2 = edge_attr2
    data.atom_subgraph = torch.tensor(atom_subgraph, dtype=torch.long)

    print(f"edge_index :{data.edge_index}")
    print(f"edge_index2 :{data.edge_index2}")
    print(f"atom_subgraph:{data.atom_subgraph}")

    return data


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.num_chirality_tag = 3
        self.num_bond_direction = 3

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # ----------- graphMAE -----------
        atom_type = F.one_hot(
            data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(
            data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(
                data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(
                data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat(
                (edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)

        # print(batch)
        # print(batch.x)

        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
        self._transform = MaskAtom(
            num_atom_type=119, num_edge_type=5, mask_rate=mask_rate, mask_edge=mask_edge)
        super(DataLoaderMaskingPred, self).__init__(
            dataset, batch_size, shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        return BatchMasking.from_data_list(batchs)


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):

        # print(self.raw_dir)
        # input()
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'zinc_standard_agent':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            for i in range(len(smiles_list)):
                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol != None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_simple(rdkit_mol)
                        # manually add mol id
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor(
                            [id])  # id here is zinc id value, stripped of
                        # leading zeros
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                except:
                    continue
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MoleculeDatasetWithBRICS(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, dataset='zinc_standard_agent', empty=False):
        self.dataset = dataset
        self.root = root

        logging.info("Initializing MoleculeDatasetWithBRICS...")
        super(MoleculeDatasetWithBRICS, self).__init__(
            root, transform, pre_transform)
        if not empty:
            logging.info("Loading processed data...")
            self.data, self.slices = torch.load(self.processed_paths[0])
            logging.info("Processed data loaded successfully.")

    def process(self):
        logging.info("Starting data processing...")
        data_list = []
        data_smiles_list = []

        if self.dataset == 'zinc_standard_agent':
            logging.info("Processing zinc_standard_agent dataset...")
            input_path = self.raw_paths[0]
            logging.info(f"Reading input data from {input_path}")
            input_df = pd.read_csv(input_path, sep=',',
                                   compression='gzip', dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            logging.info(f"Found {len(smiles_list)} SMILES strings.")

            for i, smiles in enumerate(smiles_list):
                logging.info(
                    f"Processing molecule {i+1}/{len(smiles_list)}: {smiles}")
                try:
                    rdkit_mol = Chem.MolFromSmiles(smiles)
                    if rdkit_mol:
                        logging.info(
                            "Valid molecule parsed. Converting to graph...")
                        data = mol_to_graph_data_obj_with_brics(smiles)
                        print("Atom Subgraph Indices:", data.atom_subgraph)

                        # Add ZINC ID
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor([id], dtype=torch.long)
                        logging.info(f"Added molecule ID: {id}")

                        data_list.append(data)
                        data_smiles_list.append(smiles)
                    else:
                        logging.warning(
                            f"Invalid SMILES string at index {i}: {smiles}")
                except Exception as e:
                    logging.error(f"Error processing molecule {i}: {e}")
                    continue
        else:
            raise ValueError('Invalid dataset name')

        # Pre-filter and pre-transform
        if self.pre_filter is not None:
            logging.info("Applying pre_filter...")
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            logging.info("Applying pre_transform...")
            data_list = [self.pre_transform(data) for data in data_list]

        # Save SMILES
        logging.info("Saving SMILES strings to processed directory...")
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(
            self.processed_dir, 'smiles.csv'), index=False, header=False)

        # Save data
        logging.info("Collating and saving processed data...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logging.info("Data processing complete.")

    @property
    def raw_file_names(self):
        logging.info("Retrieving raw file names...")
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        logging.info("Retrieving processed file names...")
        return 'geometric_data_with_brics.pt'

    def download(self):
        raise NotImplementedError('No download functionality for this dataset')
