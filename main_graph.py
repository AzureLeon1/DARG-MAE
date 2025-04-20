import logging
from tqdm import tqdm
import numpy as np
import os

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score


from dargmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from dargmae.datasets.data_util import load_graph_classification_dataset, load_graph_data
from dargmae.models import build_model


def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in tqdm(enumerate(dataloader)):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)
            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, args, mask_module, optimizer_mask, sub_mask_module, optimizer_sub_mask, sub_mask_weight, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g, batch_new_g, _ = batch
            batch_g = batch_g.to(device)
            batch_new_g = batch_new_g.to(device)
            feat = batch_g.ndata["attr"]
            model.train()
            mask_module.train()
            sub_mask_module.train()

            node_mask_prob = mask_module.forward(batch_g, feat, args)
            sub_mask_prob = sub_mask_module.forward(batch_new_g, feat, args)
            mask_prob = sub_mask_weight * sub_mask_prob + \
                (1 - sub_mask_weight) * node_mask_prob

            loss, loss_mask, loss_dict = model(
                batch_g, feat, epoch, args, mask_prob, pooler)

            optimizer_mask.zero_grad()
            optimizer_sub_mask.zero_grad()
            loss_mask.backward()
            optimizer_mask.step()
            optimizer_sub_mask.step()

            node_mask_prob = mask_module.forward(batch_g, feat, args)
            sub_mask_prob = sub_mask_module.forward(batch_new_g, feat, args)
            mask_prob = sub_mask_weight * sub_mask_prob + \
                (1 - sub_mask_weight) * node_mask_prob

            loss, loss_mask, loss_dict = model(
                batch_g, feat, epoch, args, mask_prob, pooler)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(
            f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f} loss_mask: {loss_mask.item():.4f}")

    return model


def collate_fn_eval(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


def collate_fn_train(batch):
    graphs = [x[0] for x in batch]
    new_graphs = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    batch_g = dgl.batch(graphs)
    batch_new_g = dgl.batch(new_graphs)

    labels = torch.cat(labels, dim=0)

    return batch_g, batch_new_g, labels


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooling = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size
    lr_mask = args.lr_mask
    lr_sub_mask = args.lr_sub_mask
    load_data = args.load_data
    sub_mask_weight = args.sub_mask_weight
    sub_avg_node_num = args.sub_avg_node_num
    sub_num = args.sub_num

    if load_data:
        current_path = os.getcwd()
        file_path = os.path.join(
            current_path, f"dargmae/datasets/data/{dataset_name}_{sub_avg_node_num}_data.pt")
        print(file_path)
        graphs, new_graphs, (num_features,
                             num_classes) = load_graph_data(file_path)
        print("Length of graphs:", len(graphs))
        print("Length of new_graphs:", len(new_graphs))
    else:
        if sub_avg_node_num is not None:
            graphs, new_graphs, (num_features, num_classes) = load_graph_classification_dataset(
                dataset_name, sub_avg_node_num=sub_avg_node_num, deg4feat=deg4feat)
        elif sub_num is not None:
            graphs, new_graphs, (num_features, num_classes) = load_graph_classification_dataset(
                dataset_name, sub_num=sub_num, deg4feat=deg4feat)
        else:
            raise NotImplementedError

    args.num_features = num_features

    combined_data = [(g, new_g, label)
                     for (g, label), (new_g, _) in zip(graphs, new_graphs)]
    print("Length of combined_data:", len(combined_data))
    train_loader = GraphDataLoader(
        combined_data, collate_fn=collate_fn_train, batch_size=batch_size, pin_memory=True, shuffle=True)
    eval_loader = GraphDataLoader(
        graphs, collate_fn=collate_fn_eval, batch_size=batch_size, shuffle=False)

    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model, mask_module, sub_mask_module = build_model(args)
        model.to(device)
        mask_module.to(device)
        sub_mask_module.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)
        optimizer_mask = create_optimizer(
            optim_type, mask_module, lr_mask, weight_decay)
        optimizer_sub_mask = create_optimizer(
            optim_type, sub_mask_module, lr_sub_mask, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")

            def scheduler(epoch): return (
                1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if not load_model:
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
                             max_epoch_f, args, mask_module, optimizer_mask, sub_mask_module, optimizer_sub_mask, sub_mask_weight, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(
            model, pooler, eval_loader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False)
        acc_list.append(test_f1)

    print(acc_list)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "results",
                             f"{dataset_name}_result.txt")

    filename = f"{dataset_name}_{args.lr}_{args.lr_mask}_{args.lr_sub_mask}_{args.sub_mask_weight}_{args.sub_avg_node_num}_{args.uniformity_dim}_{args.encoder}_{args.decoder}_{args.mask_encoder}_{args.sub_mask_encoder}_{args.lr_f}_{args.num_hidden}_{args.num_heads}_{args.num_layers}_{args.max_epoch}_{args.max_epoch_f}_{args.weight_decay}_{args.weight_decay_f}_{args.mask_rate}_{args.belta}_{args.emb_dim}_{args.lamda}_{args.alpha_0}_{args.alpha_T}_{args.gamma}_{args.in_drop}_{args.replace_rate}_{args.drop_edge_rate}_{args.alpha_l}_{args.norm}_{args.residual}_{args.scheduler}_{args.linear_prob}_{args.activation}_{args.pooling}_{args.batch_size}"

    with open(file_path, "a") as f:
        f.write(f"{filename}, {final_acc * 100:.4f}, {final_acc_std * 100:.4f}\n")
        print(f"Write results: {file_path}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
