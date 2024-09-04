import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from utils import get_metric
from model import MLP, ContrastiveLossCosine
import numpy as np


def train_(X, y, split, batch_size=1024, num_epochs=20, hidden_nodes = 512, hidden_layers = 1, lr=0.0001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Device: {device}")

    X_train, y_train = X[split], y[split]
    X_val, y_val = X[~split], y[~split]
    y_train.task, y_val.task = y.task, y.task
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

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

    val_b_acc_max = 0
    best_model = model
    best_epoch = 0
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

        if verbose: print(f"  Train: Loss={train_loss / len(train_loader):.3f}")
        model.eval()
        with torch.no_grad(): 
            y_val_pred = model.pred(X_val.to(device)).squeeze().to("cpu")
            if y.task != "all": 
                y_val_pred = y_val_pred.reshape(-1,1)
                y_val = y_val.reshape(-1,1)
                y_val.task = y.task
            # TODO: reimplement faster
            val_b_acc = sum([get_metric(y_val[:,i], y_val_pred[:,i], metric="balanced_acc") for i in range(y_val.shape[1])])/y_val.shape[1]
            #val_loss = loss_fn(model(X_val.to(device)).squeeze(), y_val.to(device)).item()
            #if verbose: print(f"  Val: Loss={val_loss:.3f}")
            #print("model: ", torch.sum(model.state_dict()['model.0.weight'], dim=[0,1]).item())
            if val_b_acc>val_b_acc_max: 
                #print("best_model: ", torch.sum(best_model.state_dict()['model.0.weight'], dim=[0,1]).item())
                val_b_acc_max = val_b_acc
                best_model = deepcopy(model)
                best_epoch = epoch 

        if verbose:
            train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
            print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
            val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
            print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")
    #print(best_epoch)
    best_model.best_epoch = best_epoch
    #print(torch.sum(best_model.state_dict()['model.0.weight'], dim=[0,1]).item())
    return best_model

def train_md(X, y, env_id, batch_size=1024, num_epochs=20, hidden_nodes = 256, hidden_layers = 1, lr=0.0001, verbose=True, ic_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Device: {device}")

    ids = np.unique(env_id)
    random_split = False
    if random_split:
        # split ids in train_ids (30%), val_ids (70%) 
        np.random.seed(42)
        np.random.shuffle(ids)
        train_ids = ids[:int(0.3*len(ids))]
        val_ids = ids[int(0.3*len(ids)):]
    else:
        # exp 1
        # train_ids = [0,1,2,3,4,5,6,7,8]
        # val_ids = [9,10,12,13,14,15,16,17, 18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        # pos 1,4
        train_ids = [1,4,10,13,19,22,28,31,37,40]
        val_ids = [0,2,3,5,6,7,8,9,12,14,15,16,17,18,20,21,23,24,25,26,27,29,30,32,33,34,35,36,38,39,41,42,43,44]

        # other
        # train_ids = [3,4,5,12,13,14,21,22,23,30,31,32,39,40,41]
        # val_ids = [1,7,10,16,19,25,28,34,37,43, 0,2,6,8,9,15,17,18,20,24,26,27,29,33,35,36,38,42,44]
        # train_ids = [4,13,22,31,40]
        # val_ids = [0,1,2,3,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,41,42,43,44]
        
    split = np.isin(env_id, train_ids)
    train_loaders = []
    for id in train_ids:
        X_train_i, y_train_i = X[env_id==id], y[env_id==id]
        train_loaders.append(DataLoader(TensorDataset(X_train_i, y_train_i), batch_size=batch_size, shuffle=True))
    X_train, y_train = X[env_id==train_ids[0]], y[env_id==train_ids[0]]
    for id in train_ids[1:]:
        X_train = torch.cat((X_train, X[env_id==id]), dim=0)
        y_train = torch.cat((y_train, y[env_id==id]), dim=0)
    y_train.task = y.task
    val_loaders = []
    for id in val_ids:
        X_val_i, y_val_i = X[env_id==id], y[env_id==id]
        val_loaders.append(DataLoader(TensorDataset(X_val_i, y_val_i), batch_size=batch_size, shuffle=True))
    X_val, y_val = X[env_id==val_ids[0]], y[env_id==val_ids[0]]
    for id in val_ids[1:]:
        X_val = torch.cat((X_val, X[env_id==id]), dim=0)
        y_val = torch.cat((y_val, y[env_id==id]), dim=0)
    y_val.task = y.task

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
    train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
    if verbose: print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
    val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
    if verbose: print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")

    # heterogenous batches
    batch_env = min([len(train_loader) for train_loader in train_loaders])
    ic_weight = ic_weight 
    #best_loss_val = np.inf
    train_metrics = []
    val_metrics = []
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        if verbose: print(f"Epoch {epoch}")
        train_loaders_iter = [iter(train_loader) for train_loader in train_loaders]
        for _ in range(batch_env):
            losses_b = []
            # loss_contr = 0
            for train_loader_iter in train_loaders_iter:
                try:
                    X_batch, y_batch = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # X_batch_pos = np.roll(X_batch, -1, axis=0).to(device)
                # X_batch_neg = np.roll(X_batch, batch_size//2, axis=0).to(device)
                # loss_contr = ContrastiveLossCosine(X_batch[:-1], X_batch_pos[:-1], torch.ones(batch_size-1, dtype=torch.float32).to(device)) + ContrastiveLossCosine(X_batch[:-1], X_batch_neg[:-1],torch.zeros(batch_size-1, dtype=torch.float32).to(device))
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss_b_e = loss_fn(y_pred, y_batch) 
                losses_b.append(loss_b_e)
            loss_b = (torch.stack(losses_b)).sum() + (torch.stack(losses_b)).var() * ic_weight
            loss_b.backward()
            optimizer.step()

        # evaluate
        # model.eval()
        # with torch.no_grad(): 
        #     losses_val = []
        #     for id in val_ids:
        #         X_val_i, y_val_i = X[env_id==id], y[env_id==id]
        #         y_val_pred_i = model.pred(X_val_i.to(device)).squeeze().to("cpu")
        #         # check if task="all"
        #         loss_val_i = loss_fn(y_val_pred_i, y_val_i.to(device))
        #         losses_val.append(loss_val_i)
        #     loss_val = (torch.stack(losses_val)).sum() + (torch.stack(losses_val)).var() * ic_weight
        #     if loss_val < best_loss_val:
        #         best_loss_val = loss_val
        #         best_model = deepcopy(model)
        #         best_model.best_epoch = epoch
        train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
        if verbose: print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
        val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
        if verbose: print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")
        train_metrics.append([train_accs, train_precisions, train_recalls])
        val_metrics.append([val_accs, val_precisions, val_recalls])
    # best_model.train_metrics = train_metrics
    # best_model.val_metrics = val_metrics
    # best_model.split = split
    model.train_metrics = train_metrics
    model.val_metrics = val_metrics
    model.split = split
    model.best_epoch = epoch
    
    return model

def train_crl(X, y, env_id, batch_size=1024, num_epochs=20, hidden_nodes = 256, hidden_layers = 1, lr=0.0001, verbose=True, ic_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Device: {device}")

    ids = np.unique(env_id)
    random_split = False
    if random_split:
        # split ids in train_ids (30%), val_ids (60%) and test_ids (10%)
        np.random.seed(42)
        np.random.shuffle(ids)
        train_ids = ids[:int(0.3*len(ids))]
        val_ids = ids[int(0.3*len(ids)):int(0.9*len(ids))]
        test_ids = ids[int(0.9*len(ids)):]
    else:
        train_ids = [0,1,2,3,4,5,6,7,8]
        val_ids = [9,10,12,13,14,15,16,17]
        test_ids = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
        
    ###### ADD DATALOADERS
    train_loaders = []
    for id in train_ids:
        X_train, y_train = X[env_id==id], y[env_id==id]
        y_train.task = y.task
        train_loaders.append(DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False))
    val_loaders = []
    for id in val_ids:
        X_val, y_val = X[env_id==id], y[env_id==id]
        y_val.task = y.task
        val_loaders.append(DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False))
    test_loaders = []
    for id in test_ids:
        X_test, y_test = X[env_id==id], y[env_id==id]
        y_test.task = y.task
        test_loaders.append(DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False))

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
    train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
    if verbose: print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
    val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
    if verbose: print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")

    # heterogenous batches
    batch_env = min([len(train_loader) for train_loader in train_loaders])
    ic_weight = ic_weight 
    val_b_acc_max = -1
    train_metrics = []
    val_metrics = []
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        if verbose: print(f"Epoch {epoch}")
        train_loaders_iter = [iter(train_loader) for train_loader in train_loaders]
        for _ in range(batch_env):
            losses_b = []
            loss_contr = 0
            for train_loader_iter in train_loaders_iter:
                try:
                    X_batch, y_batch = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch_pos = np.roll(X_batch, -1, axis=0).to(device)
                X_batch_neg = np.roll(X_batch, batch_size//2, axis=0).to(device)
                loss_contr = ContrastiveLossCosine(X_batch[:-1], X_batch_pos[:-1], torch.ones(batch_size-1, dtype=torch.float32).to(device)) + ContrastiveLossCosine(X_batch[:-1], X_batch_neg[:-1],torch.zeros(batch_size-1, dtype=torch.float32).to(device))
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss_b_e = loss_fn(y_pred, y_batch) 
                losses_b.append(loss_b_e)
            loss_b = (torch.stack(losses_b)).sum() + (torch.stack(losses_b)).var() * ic_weight
            loss_b.backward()
            optimizer.step()

        # evaluate
        model.eval()
        with torch.no_grad(): 
            y_val_pred = model.pred(X_val.to(device)).squeeze().to("cpu")
            if y.task != "all": 
                y_val_pred = y_val_pred.reshape(-1,1)
                y_val = y_val.reshape(-1,1)
                y_val.task = y.task
            # TODO: reimplement faster
            val_b_acc = sum([get_metric(y_val[:,i], y_val_pred[:,i], metric="balanced_acc") for i in range(y_val.shape[1])])/y_val.shape[1]
            if val_b_acc>val_b_acc_max: 
                val_b_acc_max = val_b_acc
                best_model = deepcopy(model)
                best_model.best_epoch = epoch 
        train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
        if verbose: print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
        val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
        if verbose: print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")
        train_metrics.append([train_accs, train_precisions, train_recalls])
        val_metrics.append([val_accs, val_precisions, val_recalls])
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
            #accs.append((y_pred.sum(dim=1) == y.sum(dim=1)).float().mean(dim=0))
            TP = ((y == 1) & (y_pred == 1)).sum(dim=0)
            FP = ((y != 1) & (y_pred == 1)).sum(dim=0)
            precisions = [prec.item() for prec in (TP / (TP + FP))]
            FN = ((y == 1) & (y_pred != 1)).sum(dim=0)
            recalls = [rec.item() for rec in (TP / (TP + FN))]
            # TN = ((y != 1) & (y_pred != 1)).sum(dim=0)
            # specificies = [spec.item() for spec in (TN / (TN + FP))]
            # accs_b = [(rec+spec)/2 for rec,spec in zip(recalls, specificies)]
        elif task in ["yellow", "blue", "or"]:
            accs = [(y_pred == y).float().mean(dim=0).item()]
            TP = ((y == 1) & (y_pred == 1)).sum()
            FP = ((y != 1) & (y_pred == 1)).sum()
            precisions = [(TP / (TP + FP)).item()]
            FN = ((y == 1) & (y_pred != 1)).sum()
            recalls = [(TP / (TP + FN)).item()]
        elif task=="sum":
            accs = [(y_pred == y).float().mean(dim=0).item()]
            precisions = []
            recalls = []
            for i in range(3):
                TP = ((y == i) & (y_pred == i)).sum()
                FP = ((y != i) & (y_pred == i)).sum()
                precisions.append((TP / (TP + FP)).item())
                FN = ((y == i) & (y_pred != i)).sum()
                recalls.append((TP / (TP + FN)).item())
    return accs, precisions, recalls

def print_performances(accs, precisions, recalls, task, environment="train"):
    if task=="all":
        print(f"  {environment}:  Accuracy=[Y2F: {accs[0]:.3f}, B2F: {accs[1]:.3f}], Precision=[Y2F {precisions[0]:.3f}, B2F {precisions[1]:.3f}], Recall=[Y2F {recalls[0]:.3f}, B2F {recalls[1]:.3f}]")
    elif task in ["yellow", "blue", "or"]:
        print(f"  {environment}:  Accuracy={accs[0]:.3f}, Precision={precisions[0]:.3f}, Recall={recalls[0]:.3f}")
    elif task=="sum":
        print(f"  {environment}:  Accuracy={accs[0]:.3f}, Precision=[0: {precisions[0]:.3f}; 1: {precisions[1]:.3f}; 2: {precisions[2]:.3f}], Recall=[0: {recalls[0]:.3f}; 1: {recalls[1]:.3f}; 2: {recalls[2]:.3f}]")
