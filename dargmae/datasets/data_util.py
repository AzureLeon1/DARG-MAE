
from collections import namedtuple, Counter
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import os

from dgl.partition import metis_partition
import dgl
import dgl.function as fn
from dgl.data import (
    load_data,
    TUDataset,
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset,
    CoraFullDataset,
    FlickrDataset,
    WikiCSDataset,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "corafull": CoraFullDataset,
    "flickr": FlickrDataset,
    "wikics": WikiCSDataset,
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def save_graph_data(file_path, dataset, new_dataset, feature_dim, num_classes):
    data_dict = {
        "dataset": dataset,
        "new_dataset": new_dataset,
        "feature_dim": feature_dim,
        "num_classes": num_classes
    }
    torch.save(data_dict, file_path)
    print(f"Data saved to {file_path}")


def load_graph_data(file_path):
    data_dict = torch.load(file_path)
    print(f"Data loaded from {file_path}")
    return data_dict["dataset"], data_dict["new_dataset"], (data_dict["feature_dim"], data_dict["num_classes"])


def save_graph_data_2(file_path, train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes):
    data_dict = {
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "eval_train_dataloader": eval_train_dataloader,
        "feature_dim": num_features,
        "num_classes": num_classes
    }
    torch.save(data_dict, file_path)
    print(f"Data saved to {file_path}")


def load_graph_data_2(file_path):
    data_dict = torch.load(file_path)
    print(f"Data loaded from {file_path}")
    return data_dict["train_dataloader"], data_dict["valid_dataloader"], data_dict["test_dataloader"], data_dict["eval_train_dataloader"], data_dict["feature_dim"], data_dict["num_classes"]


def transductive_split_to_subgraph(dataset, g, sub_avg_node_num=None, sub_num=None):

    print(g)
    print(f"Number of nodes: {g.num_nodes()}")
    if sub_avg_node_num is not None:
        best_num_subgraphs = g.num_nodes() // sub_avg_node_num
        print(f"Optimal number of subgraphs: {best_num_subgraphs}")
    elif sub_num is not None:
        best_num_subgraphs = sub_num
        print(f"Optimal number of subgraphs: {best_num_subgraphs}")
    else:
        print("sub_avg_node_num and sub_num cannot be None at the same time")
    partition = dgl.metis_partition(g, k=best_num_subgraphs)

    subgraph_id = [-1] * g.num_nodes()

    for part_id, subgraph in partition.items():
        node_ids = subgraph.ndata['_ID'].tolist()
        for node_id in node_ids:
            subgraph_id[node_id] = part_id

    subgraph_ids = torch.tensor(subgraph_id, dtype=torch.int32)

    g.ndata['subgraph_id'] = subgraph_ids + 1
    print("subgraph_ids", g.ndata['subgraph_id'])

    src, dst = g.edges()
    mask = subgraph_ids[src] == subgraph_ids[dst]

    new_graph = dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes())
    new_graph.ndata.update(g.ndata)
    if dataset == "cora" or dataset == "corafull" or dataset == "wikics":
        print(f"Save the edges of {dataset}...")
        new_graph.edata['__orig__'] = g.edata['__orig__'][mask]
    elif dataset == "reddit":
        print(f"Save the edges of {dataset}...")
        new_graph.edata['__orig__'] = g.edata['__orig__'][mask]
        new_graph.edata['_ID'] = g.edata['_ID'][mask]

    return new_graph


def clean_graph(g):
    g = dgl.to_simple(g)
    return g


def graph_split_to_subgraph(dataset, sub_avg_node_num=None, sub_num=None):
    new_dataset = []
    global_subgraph_id = 0

    for g, label in tqdm(dataset):
        g = clean_graph(g)

        print(g)

        print(f"Number of nodes: {g.num_nodes()}")
        if sub_avg_node_num is not None:
            best_num_subgraphs = max(2, g.num_nodes() // sub_avg_node_num)
            print(f"Optimal number of subgraphs: {best_num_subgraphs}")
        elif sub_num is not None:
            best_num_subgraphs = sub_num
            print(f"Optimal number of subgraphs: {best_num_subgraphs}")
        else:
            print("sub_avg_node_num and sub_num cannot be None at the same time")

        try:
            partition = dgl.metis_partition(g, k=best_num_subgraphs)
        except Exception as e:
            print(f"Metis partition failed: {e}")
            # print("="*600)
            continue

        subgraph_id = [-1] * g.num_nodes()
        for part_id, subgraph in partition.items():
            node_ids = subgraph.ndata['_ID'].tolist()
            for node_id in node_ids:
                subgraph_id[node_id] = part_id

        subgraph_ids = torch.tensor(subgraph_id, dtype=torch.int32)
        g.ndata['subgraph_id'] = subgraph_ids + 1 + global_subgraph_id
        global_subgraph_id += best_num_subgraphs
        print(f"subgraph_ids: {g.ndata['subgraph_id']}")

        src, dst = g.edges()
        mask = subgraph_ids[src] == subgraph_ids[dst]
        if mask.sum() == 0:
            print(f"Warning: Some of the graphs in graph {g} have no edges.")
            # print("="*600)
            continue

        new_g = dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes())
        new_g.ndata.update(g.ndata)
        if 'count' in g.edata:
            print("Save Edge...")
            new_g.edata['count'] = g.edata['count'][mask]

        new_dataset.append((new_g, label))

    return new_dataset


def inductive_split_to_subgraph(dataset_name, dataset, sub_avg_node_num=None, sub_num=None):
    global_subgraph_id = 0
    if dataset_name == "ppi":
        new_dataset = []
        for g in tqdm(dataset):
            print(f"Number of nodes: {g.num_nodes()}")

            print(g)

            if sub_avg_node_num is not None:
                best_num_subgraphs = g.num_nodes() // sub_avg_node_num
                print(f"Optimal number of subgraphs: {best_num_subgraphs}")
            elif sub_num is not None:
                best_num_subgraphs = sub_num
                print(f"Optimal number of subgraphs: {best_num_subgraphs}")
            else:
                print("sub_avg_node_num and sub_num cannot be None at the same time")

            partition = dgl.metis_partition(g, k=best_num_subgraphs)

            subgraph_id = [-1] * g.num_nodes()

            for part_id, subgraph in partition.items():
                node_ids = subgraph.ndata['_ID'].tolist()
                for node_id in node_ids:
                    subgraph_id[node_id] = part_id

            subgraph_ids = torch.tensor(subgraph_id, dtype=torch.int32)

            g.ndata['subgraph_id'] = subgraph_ids + 1 + global_subgraph_id
            global_subgraph_id += best_num_subgraphs
            print(f"subgraph_ids: {g.ndata['subgraph_id']}")

            src, dst = g.edges()
            mask = subgraph_ids[src] == subgraph_ids[dst]

            new_g = dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes())
            new_g.ndata.update(g.ndata)

            print("Save ppi edge...")
            new_g.edata['_ID'] = g.edata['_ID'][mask]

            new_dataset.append(new_g)

    return new_dataset


def load_dataset(dataset_name, sub_avg_node_num=None, sub_num=None):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full(
            (num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full(
            (num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full(
            (num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        labels = graph.ndata["label"]
        if dataset_name in ["corafull"]:
            train_idx, test_idx, train_labels, test_labels = train_test_split(
                np.arange(graph.number_of_nodes()),
                labels,
                test_size=0.6)

            train_idx, val_idx, train_labels, val_labels = train_test_split(
                train_idx,
                train_labels,
                test_size=0.75)
            train_mask = torch.BoolTensor(
                [idx in train_idx for idx in range(len(labels))])
            val_mask = torch.BoolTensor(
                [idx in val_idx for idx in range(len(labels))])
            test_mask = torch.BoolTensor(
                [idx in test_idx for idx in range(len(labels))])
            # Set node features
            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes

    if sub_avg_node_num is not None:
        new_graph = transductive_split_to_subgraph(
            dataset_name, graph, sub_avg_node_num=sub_avg_node_num)
    elif sub_num is not None:
        new_graph = transductive_split_to_subgraph(
            dataset_name, graph, sub_num=sub_num)
    else:
        print("sub_avg_node_num and sub_num cannot be None at the same time")

    current_path = os.getcwd()
    file_path = os.path.join(
        current_path, f"dargmae/datasets/data/{dataset_name}_{sub_avg_node_num}_data.pt")
    save_graph_data(file_path, graph, new_graph, num_features, num_classes)

    return graph, new_graph, (num_features, num_classes)


def collate_fn(batch):
    graphs, new_graphs = zip(*batch)

    batch_g = dgl.batch(graphs)
    batch_new_g = dgl.batch(new_graphs)

    assert batch_g.num_nodes() == batch_new_g.num_nodes(), "Node count does not match!"

    return batch_g, batch_new_g


def load_inductive_dataset(dataset_name, sub_avg_node_num=None, sub_num=None):
    if dataset_name == "ppi":
        batch_size = 2
        train_dataset = PPIDataset(mode='train')
        print("train_dataset:", train_dataset)
        if sub_avg_node_num is not None:
            new_train_dataset = inductive_split_to_subgraph(
                dataset_name, train_dataset, sub_avg_node_num=sub_avg_node_num)
        elif sub_num is not None:
            new_train_dataset = inductive_split_to_subgraph(
                dataset_name, train_dataset, sub_num=sub_num)
        else:
            print("sub_avg_node_num and sub_num cannot be None at the same time")
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        combined_data = list(zip(train_dataset, new_train_dataset))
        train_dataloader = GraphDataLoader(
            combined_data, collate_fn=collate_fn, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]

    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)

        print(f"g: {g}")
        print(f"train_g: {train_g}")
        if sub_avg_node_num is not None:
            sub_train_g = transductive_split_to_subgraph(
                dataset_name, train_g, sub_avg_node_num=sub_avg_node_num)
        elif sub_num is not None:
            sub_train_g = transductive_split_to_subgraph(
                dataset_name, train_g, sub_num=sub_num)
        else:
            print("sub_avg_node_num and sub_num cannot be None at the same time")

        train_dataloader = [train_g, sub_train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]

    current_path = os.getcwd()
    file_path = os.path.join(
        current_path, f"dargmae/datasets/data/{dataset_name}_{sub_avg_node_num}_data.pt")
    save_graph_data_2(file_path, train_dataloader, valid_dataloader,
                      test_dataloader, eval_train_dataloader, num_features, num_classes)

    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes


def load_graph_classification_dataset(dataset_name, sub_avg_node_num=None, sub_num=None, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(
                    feature_dim, g.ndata["node_labels"].max().item())

            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES

                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])

    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    if sub_avg_node_num is not None:
        new_dataset = graph_split_to_subgraph(
            dataset, sub_avg_node_num=sub_avg_node_num)
    elif sub_num is not None:
        new_dataset = graph_split_to_subgraph(dataset, sub_num=sub_num)
    else:
        print("sub_avg_node_num and sub_num cannot be None at the same time")

    print(
        f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    current_path = os.getcwd()
    file_path = os.path.join(
        current_path, f"dargmae/datasets/data/{dataset_name}_{sub_avg_node_num}_data.pt")
    save_graph_data(file_path, dataset, new_dataset, feature_dim, num_classes)

    return dataset, new_dataset, (feature_dim, num_classes)
