from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool


def build_adv(args):
    mask_module = AdversMask(args)
    return mask_module


def build_sub_adv(args):
    mask_module = Sub_AdversMask(args)
    return mask_module


class AdversMask(nn.Module):
    def __init__(self, args):
        super(AdversMask, self).__init__()
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # device = torch.device("cpu")
        self.mask = GNN(
            args.num_layer,
            args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        ).to(device)
        ## self.mask = GNN(args.num_layer_node, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)####
        self.fc_mlp = nn.Linear(args.emb_dim, 2).to(device)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        gnn_emb = self.mask(x, edge_index, edge_attr)
        z = F.gumbel_softmax(self.fc_mlp(gnn_emb), hard=True)
        return z


class Sub_AdversMask(nn.Module):
    def __init__(self, args):
        super(Sub_AdversMask, self).__init__()
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        self.mask = GNN(
            args.num_layer,
            args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        ).to(device)

        self.subgraph_fc = nn.Linear(args.emb_dim, 2).to(device)

    def forward(self, batch):
        """
        :param x: Atom node features, shape (num_nodes, emb_dim)
        :param edge_index: Edge indices
        :param edge_attr: Edge features
        :param atom_subgraph: Subgraph index for each atom, shape (num_nodes,)
        :return: Retention/non-retention probabilities of atom nodes, shape (num_nodes, 2)
        """
        x = batch.x
        edge_index = batch.edge_index2
        edge_attr = batch.edge_attr2

        atom_subgraph = batch.atom_subgraph

        gnn_emb = self.mask(x, edge_index, edge_attr)  # (num_nodes, emb_dim)

        unique_subgraph, new_atom_subgraph = torch.unique(
            atom_subgraph, return_inverse=True
        )

        subgraph_emb = global_add_pool(gnn_emb, new_atom_subgraph)

        subgraph_logits = self.subgraph_fc(subgraph_emb)  # (num_subgraphs, 2)
        subgraph_prob = F.gumbel_softmax(
            subgraph_logits, hard=True
        )  # (num_subgraphs, 2)

        node_prob = subgraph_prob[new_atom_subgraph]

        return node_prob


num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(
            edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

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

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
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
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

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

        return node_representation
