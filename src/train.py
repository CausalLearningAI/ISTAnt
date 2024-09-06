import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from model import MLP
import numpy as np


def train_(supervised, batch_size=1024, num_epochs=20, hidden_nodes = 512, hidden_layers = 1, lr=0.0001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Device: {device}")

    # get training and validation set
    X = supervised["X"]
    y = supervised["Y"]
    split = supervised["split"]
    X_train, y_train = X[split], y[split]
    X_val, y_val = X[~split], y[~split]
    y_train.task, y_val.task = y.task, y.task
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # get model
    input_size = X.shape[1]
    model = MLP(input_size, hidden_nodes, hidden_layers, task=y.task).to(device)
    model.device = device
    model.task = y.task
    model.token = X.token
    model.encoder = X.encoder_name
    if y.task == "sum":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        pos_weight = ((y_train==0).sum(dim=0)/(y_train==1).sum(dim=0)).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose: print("Starting perfomances")
    train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
    if verbose: print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
    val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
    if verbose: print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")

    best_val_bacc = 0
    train_metrics = []
    val_metrics = []
    for epoch in range(1, num_epochs+1):
        if verbose: print(f"Epoch {epoch}")
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if y.task=="sum":
                y_batch = y_batch.long()
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = loss_fn(y_pred, y_batch) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if verbose: print(f"Epoch {epoch} (train loss: {train_loss/len(train_loader):.3f})")

        # evaluate
        train_accs, train_baccs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
        if verbose: print_performances(train_accs, train_baccs, train_precisions, train_recalls, y.task, environment="train")
        val_accs, val_baccs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
        if verbose: print_performances(val_accs, val_baccs, val_precisions, val_recalls, y.task, environment="val")
        if np.mean(val_baccs) >= best_val_bacc:
                best_val_bacc = np.mean(val_baccs)
                best_model = deepcopy(model)
                best_model.best_epoch = epoch

        train_metrics.append([train_accs, train_baccs, train_precisions, train_recalls])
        val_metrics.append([val_accs, val_baccs, val_precisions, val_recalls])

    best_model.train_metrics = train_metrics
    best_model.val_metrics = val_metrics

    return best_model

def train_md(supervised, batch_size=1024, num_epochs=20, hidden_nodes = 256, hidden_layers = 1, lr=0.0001, verbose=True, ic_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Device: {device}")

    # get training and validation set
    X = supervised["X"]
    y = supervised["Y"]
    E = supervised["E"]
    split = supervised["split"]
    envs_train = np.unique(E[split])
    envs_val = np.unique(E[~split])

    train_loaders = []
    X_train, y_train = X[E==envs_train[0]], y[E==envs_train[0]]
    train_loaders.append(DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True))
    for env in envs_train[1:]:
        X_train_i, y_train_i = X[E==env], y[E==env]
        train_loaders.append(DataLoader(TensorDataset(X_train_i, y_train_i), batch_size=batch_size, shuffle=True))
        X_train = torch.cat((X_train, X_train_i), dim=0)
        y_train = torch.cat((y_train, y_train_i), dim=0)
    y_train.task = y.task

    val_loaders = []
    X_val, y_val = X[E==envs_val[0]], y[E==envs_val[0]]
    val_loaders.append(DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True))
    for env in envs_val[1:]:
        X_val_i, y_val_i = X[E==env], y[E==env]
        val_loaders.append(DataLoader(TensorDataset(X_val_i, y_val_i), batch_size=batch_size, shuffle=True))
        X_val = torch.cat((X_val, X_val_i), dim=0)
        y_val = torch.cat((y_val, y_val_i), dim=0)
    y_val.task = y.task

    # get model
    input_size = X.shape[1]
    model = MLP(input_size, hidden_nodes=hidden_nodes, hidden_layers=hidden_layers, task="rep").to(device)
    model.device = device
    model.task = y.task
    model.token = X.token
    model.encoder = X.encoder_name
    if y.task == "sum":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        pos_weight = ((y==0).sum(dim=0)/(y==1).sum(dim=0)).to(device) # to fix (independet to test set)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose: print('Train Heterogenous, IC Weight:', ic_weight)

    if verbose: print("Starting perfomances")
    train_accs, train_baccs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
    if verbose: print_performances(train_accs, train_baccs, train_precisions, train_recalls, y.task, environment="train")
    val_accs, val_baccs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
    if verbose: print_performances(val_accs,val_baccs, val_precisions, val_recalls, y.task, environment="val")

    # heterogenous batches
    batch_env = min([len(train_loader) for train_loader in train_loaders])
    ic_weight = ic_weight 
    best_val_bacc = 0
    train_metrics = []
    val_metrics = []
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        train_loss = 0
        train_loaders_iter = [iter(train_loader) for train_loader in train_loaders]
        for _ in range(batch_env):
            losses_b = []
            for train_loader_iter in train_loaders_iter:
                try:
                    X_batch, y_batch = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss_b_e = loss_fn(y_pred, y_batch) 
                losses_b.append(loss_b_e)
            loss_b = (torch.stack(losses_b)).sum() + (torch.stack(losses_b)).var() * ic_weight
            loss_b.backward()
            optimizer.step()
            train_loss += loss_b.item()
        if verbose: print(f"Epoch {epoch} (train loss: {train_loss/batch_env:.3f})")

        # evaluate
        train_accs, train_baccs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
        if verbose: print_performances(train_accs, train_baccs, train_precisions, train_recalls, y.task, environment="train")
        val_accs, val_baccs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
        if verbose: print_performances(val_accs, val_baccs, val_precisions, val_recalls, y.task, environment="val")
        if np.mean(val_baccs) >= best_val_bacc:
                best_val_bacc = np.mean(val_baccs)
                best_model = deepcopy(model)
                best_model.best_epoch = epoch

        train_metrics.append([train_accs, train_baccs, train_precisions, train_recalls])
        val_metrics.append([val_accs, val_baccs, val_precisions, val_recalls])

    best_model.train_metrics = train_metrics
    best_model.val_metrics = val_metrics
    
    return best_model

def evaluate_model(model, X, y, device="cpu"):
    task = y.task
    model.eval()
    with torch.no_grad():
        y_pred = model.pred(X.to(device)).to("cpu").squeeze()
        y = y.squeeze()
        if task=="all":
            accs = [acc.item() for acc in (y_pred == y).float().mean(dim=0)]
            TP = ((y == 1) & (y_pred == 1)).sum(dim=0)
            FP = ((y != 1) & (y_pred == 1)).sum(dim=0)
            precisions = [prec.item() for prec in (TP / (TP + FP))]
            FN = ((y == 1) & (y_pred != 1)).sum(dim=0)
            recalls = [rec.item() for rec in (TP / (TP + FN))]
            TN = ((y != 1) & (y_pred != 1)).sum(dim=0)
            specificies = [spec.item() for spec in (TN / (TN + FP))]
            baccs = [(rec+spec)/2 for rec,spec in zip(recalls, specificies)]
        elif task in ["yellow", "blue", "or"]:
            accs = [(y_pred == y).float().mean(dim=0).item()]
            TP = ((y == 1) & (y_pred == 1)).sum()
            FP = ((y != 1) & (y_pred == 1)).sum()
            precisions = [(TP / (TP + FP)).item()]
            FN = ((y == 1) & (y_pred != 1)).sum()
            recalls = [(TP / (TP + FN)).item()]
            TN = ((y != 1) & (y_pred != 1)).sum()
            specificies = [(TN / (TN + FP)).item()]
            baccs = [(recalls[0]+specificies[0])/2]
        elif task=="sum":
            accs = [(y_pred == y).float().mean(dim=0).item()]
            precisions = []
            recalls = []
            baccs = []
            for i in range(3):
                TP = ((y == i) & (y_pred == i)).sum()
                FP = ((y != i) & (y_pred == i)).sum()
                precisions.append((TP / (TP + FP)).item())
                FN = ((y == i) & (y_pred != i)).sum()
                recalls.append((TP / (TP + FN)).item())
                TN = ((y != i) & (y_pred != i)).sum()
                specificies = (TN / (TN + FP)).item()
                baccs.append((recalls[i]+specificies)/2)
    return accs, baccs, precisions, recalls

def print_performances(accs, baccs, precisions, recalls, task, environment="train"):
    if task=="all":
        print(f"  {environment}:  Accuracy=[Y2F: {accs[0]:.3f}, B2F: {accs[1]:.3f}], Balanced Accuracy=[Y2F: {baccs[0]:.3f}, B2F: {baccs[1]:.3f}], Precision=[Y2F {precisions[0]:.3f}, B2F {precisions[1]:.3f}], Recall=[Y2F {recalls[0]:.3f}, B2F {recalls[1]:.3f}]")
    elif task in ["yellow", "blue", "or"]:
        print(f"  {environment}:  Accuracy={accs[0]:.3f}, Balanced Accuracy={baccs[0]:.3f},  Precision={precisions[0]:.3f}, Recall={recalls[0]:.3f}")
    elif task=="sum":
        print(f"  {environment}:  Accuracy={accs[0]:.3f}, Balanced Accuracy={baccs[0]:.3f}, Precision=[0: {precisions[0]:.3f}; 1: {precisions[1]:.3f}; 2: {precisions[2]:.3f}], Recall=[0: {recalls[0]:.3f}; 1: {recalls[1]:.3f}; 2: {recalls[2]:.3f}]")
