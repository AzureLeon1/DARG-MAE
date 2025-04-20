import dgl
import logging
import numpy as np
from tqdm import tqdm
import torch
import os

from dargmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from dargmae.datasets.data_util import load_dataset, load_graph_data
from dargmae.evaluation import node_classification_evaluation
from dargmae.models import build_model


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def var(graph, new_graph):
    graph_feats = graph.ndata["feat"]
    new_graph_feats = new_graph.ndata["feat"]
    if graph.num_nodes() == new_graph.num_nodes():
        if torch.allclose(graph_feats, new_graph_feats):
            print("One-to-one correspondence between node features")
        else:
            print("Inconsistent node characteristics")
    else:
        print("The number of nodes in the graph is inconsistent")


def pretrain(
    model,
    graph,
    new_graph,
    feat,
    optimizer,
    max_epoch,
    device,
    scheduler,
    num_classes,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    linear_prob,
    args,
    mask_module,
    optimizer_mask,
    sub_mask_module,
    optimizer_sub_mask,
    sub_mask_weight,
    logger=None,
):
    logging.info("start training..")
    graph = graph.to(device)
    new_graph = new_graph.to(device)
    x = feat.to(device)
    var(graph, new_graph)

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        mask_module.train()
        sub_mask_module.train()

        node_mask_prob = mask_module.forward(graph, x, args)
        sub_mask_prob = sub_mask_module.forward(new_graph, x, args)
        mask_prob = (
            sub_mask_weight * sub_mask_prob +
            (1 - sub_mask_weight) * node_mask_prob
        )

        loss, loss_mask, loss_dict = model(
            graph, x, epoch, args, mask_prob, 0.0)

        optimizer_mask.zero_grad()
        optimizer_sub_mask.zero_grad()
        loss_mask.backward()
        optimizer_mask.step()
        optimizer_sub_mask.step()

        node_mask_prob = mask_module.forward(graph, x, args)
        sub_mask_prob = sub_mask_module.forward(new_graph, x, args)
        mask_prob = (
            sub_mask_weight * sub_mask_prob +
            (1 - sub_mask_weight) * node_mask_prob
        )

        loss, loss_mask, loss_dict = model(
            graph, x, epoch, args, mask_prob, 0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(
            f"# Epoch {epoch}: train_loss: {loss.item():.4f} loss_mask: {loss_mask.item():.4f}"
        )
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
    return model


def main(args):
    device = args.device
    seeds = args.seeds
    print(seeds)
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
    lr_mask = args.lr_mask
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    lr_sub_mask = args.lr_sub_mask
    load_data = args.load_data
    sub_mask_weight = args.sub_mask_weight
    sub_avg_node_num = args.sub_avg_node_num
    sub_num = args.sub_num

    print("load_data", load_data)
    if load_data:
        current_path = os.getcwd()
        file_path = os.path.join(
            current_path,
            f"dargmae/datasets/data/{dataset_name}_{sub_avg_node_num}_data.pt",
        )
        print(file_path)
        graph, new_graph, (num_features,
                           num_classes) = load_graph_data(file_path)
    else:
        if sub_avg_node_num is not None:
            graph, new_graph, (num_features, num_classes) = load_dataset(
                dataset_name, sub_avg_node_num=sub_avg_node_num
            )
        elif sub_num is not None:
            graph, new_graph, (num_features, num_classes) = load_dataset(
                dataset_name, sub_num=sub_num
            )
        else:
            raise NotImplementedError

    args.num_features = num_features
    args.max_degree = graph.in_degrees().max() + 1

    acc_list = []
    estp_acc_list = []
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        if dataset_name == "wikics":
            graph.ndata["train_mask"] = train_mask[:, i]
            graph.ndata["val_mask"] = val_mask[:, i]
        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}"
            )
        else:
            logger = None

        model, mask_module, sub_mask_module = build_model(args)
        model.to(device)
        mask_module.to(device)
        sub_mask_module.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)
        optimizer_mask = create_optimizer(
            optim_type, mask_module, lr_mask, weight_decay
        )
        optimizer_sub_mask = create_optimizer(
            optim_type, sub_mask_module, lr_sub_mask, weight_decay
        )

        if use_scheduler:
            logging.info("Use schedular")

            def scheduler(epoch): return (
                1 + np.cos(epoch * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=scheduler
            )
        else:
            scheduler = None

        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(
                model,
                graph,
                new_graph,
                x,
                optimizer,
                max_epoch,
                device,
                scheduler,
                num_classes,
                lr_f,
                weight_decay_f,
                max_epoch_f,
                linear_prob,
                args,
                mask_module,
                optimizer_mask,
                sub_mask_module,
                optimizer_sub_mask,
                sub_mask_weight,
                logger,
            )
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(
            model,
            graph,
            x,
            num_classes,
            lr_f,
            weight_decay_f,
            max_epoch_f,
            device,
            linear_prob,
        )
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    print(f"# final_acc: {acc_list}")
    print(f"# early-stopping_acc: {estp_acc_list}")

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.8f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.8f}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "results",
                             f"{dataset_name}_result.txt")

    filename = f"{dataset_name}_{args.lr}_{args.lr_mask}_{args.lr_sub_mask}_{args.sub_mask_weight}_{args.sub_avg_node_num}_{args.uniformity_dim}_{args.encoder}_{args.decoder}_{args.mask_encoder}_{args.sub_mask_encoder}_{args.lr_f}_{args.num_hidden}_{args.num_heads}_{args.num_layers}_{args.max_epoch}_{args.max_epoch_f}_{args.weight_decay}_{args.weight_decay_f}_{args.mask_rate}_{args.belta}_{args.emb_dim}_{args.lamda}_{args.alpha_0}_{args.alpha_T}_{args.gamma}_{args.in_drop}_{args.replace_rate}_{args.drop_edge_rate}_{args.alpha_l}_{args.norm}_{args.residual}_{args.scheduler}_{args.linear_prob}_{args.activation}_{args.pooling}_{args.batch_size}"

    with open(file_path, "a") as f:
        f.write(
            f"{filename}, {final_acc * 100:.2f}, {final_acc_std * 100:.2f}, {estp_acc * 100:.2f}, {estp_acc_std * 100:.2f}\n"
        )
        print(f"Write results: {file_path}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
