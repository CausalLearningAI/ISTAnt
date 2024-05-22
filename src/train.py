import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from utils import get_metric
from model import MLP

def train_model(X, y, split, batch_size=1024, num_epochs=20, hidden_nodes = 512, hidden_layers = 1, lr=0.0001, verbose=True):
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
