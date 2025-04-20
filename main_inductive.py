
import numpy as np
import torch
from sklearn.metrics import f1_score

import logging
import yaml
import numpy as np
from tqdm import tqdm
import torch
import os
import time

from dargmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
)
from dargmae.datasets.data_util import load_inductive_dataset, load_graph_data_2, collate_fn
from dargmae.models import build_model
from dargmae.evaluation import linear_probing_for_inductive_node_classiifcation, LogisticRegression


def evaluete(model, loaders, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        if len(loaders[0]) > 1:
            x_all = {"train": [], "val": [], "test": []}
            y_all = {"train": [], "val": [], "test": []}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        x_all[key].append(x)
                        y_all[key].append(subgraph.ndata["label"])
            in_dim = x_all["train"][0].shape[1]
            encoder = LogisticRegression(in_dim, num_classes)
            num_finetune_params = [p.numel()
                                   for p in encoder.parameters() if p.requires_grad]
            if not mute:
                print(
                    f"num parameters for finetuning: {sum(num_finetune_params)}")
                # torch.save(x.cpu(), "feat.pt")

            encoder.to(device)
            optimizer_f = create_optimizer(
                "adam", encoder, lr_f, weight_decay_f)
            final_acc, estp_acc = mutli_graph_linear_evaluation(
                encoder, x_all, y_all, optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc
        else:
            x_all = {"train": None, "val": None, "test": None}
            y_all = {"train": None, "val": None, "test": None}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        mask = subgraph.ndata[f"{key}_mask"]
                        x_all[key] = x[mask]
                        y_all[key] = subgraph.ndata["label"][mask]
            in_dim = x_all["train"].shape[1]

            encoder = LogisticRegression(in_dim, num_classes)
            encoder = encoder.to(device)
            optimizer_f = create_optimizer(
                "adam", encoder, lr_f, weight_decay_f)

            x = torch.cat(list(x_all.values()))
            y = torch.cat(list(y_all.values()))
            num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
            num_nodes = num_train + num_val + num_test
            train_mask = torch.arange(num_train, device=device)
            val_mask = torch.arange(
                num_train, num_train + num_val, device=device)
            test_mask = torch.arange(
                num_train + num_val, num_nodes, device=device)

            final_acc, estp_acc = linear_probing_for_inductive_node_classiifcation(
                encoder, x, y, (train_mask, val_mask, test_mask), optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc
    else:
        raise NotImplementedError


def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            test_out = np.where(test_out >= 0, 1, 0)

            val_acc = f1_score(val_label, val_out, average="micro")
            test_acc = f1_score(test_label, test_out, average="micro")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    if mute:
        print(
            f"# IGNORE: --- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f},  Final-TestAcc: {test_acc:.4f}--- ")
    else:
        print(
            f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- ")

    return test_acc, best_val_test_acc


def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, args, mask_module, optimizer_mask, sub_mask_module, optimizer_sub_mask, sub_mask_weight, logger=None):
    logging.info("start training..")
    train_loader, val_loader, test_loader, eval_train_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))

    if isinstance(train_loader, list) and len(train_loader) == 2:
        sub_train_loader = [train_loader[1].to(device)]
        train_loader = [train_loader[0].to(device)]
        eval_train_loader = train_loader
        eval_sub_train_loader = sub_train_loader
        train_loader = zip(train_loader, sub_train_loader)
    if isinstance(val_loader, list) and len(val_loader) == 1:
        val_loader = [val_loader[0].to(device)]
        test_loader = val_loader

    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for subgraph, sub_subgraph in train_loader:
            subgraph = subgraph.to(device)
            sub_subgraph = sub_subgraph.to(device)

            model.train()
            mask_module.train()
            sub_mask_module.train()

            node_mask_prob = mask_module.forward(
                subgraph, subgraph.ndata["feat"], args)
            sub_mask_prob = sub_mask_module.forward(
                sub_subgraph, sub_subgraph.ndata["feat"], args)
            mask_prob = sub_mask_weight * sub_mask_prob + \
                (1 - sub_mask_weight) * node_mask_prob

            loss, loss_mask, loss_dict = model(
                subgraph, subgraph.ndata["feat"], epoch, args, mask_prob, 0.)

            optimizer_mask.zero_grad()
            optimizer_sub_mask.zero_grad()
            loss_mask.backward()
            optimizer_mask.step()
            optimizer_sub_mask.step()

            node_mask_prob = mask_module.forward(
                subgraph, subgraph.ndata["feat"], args)
            sub_mask_prob = sub_mask_module.forward(
                sub_subgraph, sub_subgraph.ndata["feat"], args)
            mask_prob = sub_mask_weight * sub_mask_prob + \
                (1 - sub_mask_weight) * node_mask_prob

            loss, loss_mask, loss_dict = model(
                subgraph, subgraph.ndata["feat"], epoch, args, mask_prob, 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(
            f"# Epoch {epoch} | train_loss: {train_loss:.4f} loss_mask: {loss_mask.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    return model


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
    lr_mask = args.lr_mask
    lr_sub_mask = args.lr_sub_mask
    load_data = args.load_data
    sub_mask_weight = args.sub_mask_weight
    sub_avg_node_num = args.sub_avg_node_num
    sub_num = args.sub_num

    print(args.load_data)
    if load_data:
        current_path = os.getcwd()
        file_path = os.path.join(
            current_path, f"dargmae/datasets/data/{dataset_name}_{sub_avg_node_num}_data.pt")
        print(file_path)
        (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            eval_train_dataloader,
            num_features,
            num_classes
        ) = load_graph_data_2(file_path)

    else:
        if sub_avg_node_num is not None:
            (
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                eval_train_dataloader,
                num_features,
                num_classes
            ) = load_inductive_dataset(dataset_name, sub_avg_node_num=sub_avg_node_num)
        elif sub_num is not None:
            (
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                eval_train_dataloader,
                num_features,
                num_classes
            ) = load_inductive_dataset(dataset_name, sub_num=sub_num)
        else:
            raise NotImplementedError

    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
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
            model = pretrain(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes,
                             lr_f, weight_decay_f, max_epoch_f, linear_prob, args, mask_module, optimizer_mask, sub_mask_module, optimizer_sub_mask, sub_mask_weight, logger)
        model = model.cpu()

        model = model.to(device)
        model.eval()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = evaluete(model, (eval_train_dataloader, valid_dataloader, test_dataloader),
                                       num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    print(acc_list)
    print(estp_acc_list)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, es_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_f1: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_f1: {estp_acc:.4f}±{es_acc_std:.4f}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "results",
                             f"{dataset_name}_result.txt")

    filename = f"{dataset_name}_{args.lr}_{args.lr_mask}_{args.lr_sub_mask}_{args.sub_mask_weight}_{args.sub_avg_node_num}_{args.uniformity_dim}_{args.encoder}_{args.decoder}_{args.mask_encoder}_{args.sub_mask_encoder}_{args.lr_f}_{args.num_hidden}_{args.num_heads}_{args.num_layers}_{args.max_epoch}_{args.max_epoch_f}_{args.weight_decay}_{args.weight_decay_f}_{args.mask_rate}_{args.belta}_{args.emb_dim}_{args.lamda}_{args.alpha_0}_{args.alpha_T}_{args.gamma}_{args.in_drop}_{args.replace_rate}_{args.drop_edge_rate}_{args.alpha_l}_{args.norm}_{args.residual}_{args.scheduler}_{args.linear_prob}_{args.activation}_{args.pooling}_{args.batch_size}"

    with open(file_path, "a") as f:
        f.write(f"{filename}, {final_acc * 100:.2f}, {final_acc_std * 100:.2f}, {estp_acc * 100:.2f}, {es_acc_std * 100:.2f}\n")
        print(f"Write results: {file_path}")


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
