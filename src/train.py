import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import MLP


def train_model(X, y, split, batch_size=1024, num_epochs=20, lr=0.0001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train = X[split], y[split]
    X_val, y_val = X[~split], y[~split]
    y_train.task, y_val.task = y.task, y.task
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    input_size = X.shape[1]
    hidden_size = 256

    model = MLP(input_size, hidden_size, task=y.task).to(device)
    model.device = device
    if y.task == "sum":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting perfomances")
    train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
    if verbose: print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
    val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
    if verbose: print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = loss_fn(y_pred, y_batch) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if verbose: print(f"  Train: Loss={train_loss / len(train_loader):.3f}")
        if verbose:
            train_accs, train_precisions, train_recalls = evaluate_model(model, X_train, y_train, device)
            print_performances(train_accs, train_precisions, train_recalls, y.task, environment="train")
            val_accs, val_precisions, val_recalls = evaluate_model(model, X_val, y_val, device)
            print_performances(val_accs, val_precisions, val_recalls, y.task, environment="val")

    return model

def evaluate_model(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        y_pred = model.pred(X.to(device)).to("cpu").squeeze()
        if y.task=="all":
            accs = [acc.item() for acc in (y_pred == y).float().mean(dim=0)]
            #accs.append((y_pred.sum(dim=1) == y.sum(dim=1)).float().mean(dim=0))
            TP = ((y == 1) & (y_pred == 1)).sum(dim=0)
            FP = ((y != 1) & (y_pred == 1)).sum(dim=0)
            precisions = TP / (TP + FP)
            FN = ((y == 1) & (y_pred != 1)).sum(dim=0)
            recalls = TP / (TP + FN)
        elif y.task in ["yellow", "blue", "or"]:
            accs = (y_pred == y).float().mean(dim=0).item()
            TP = ((y == 1) & (y_pred == 1)).sum()
            #print("TP: ", TP.item())
            FP = ((y != 1) & (y_pred == 1)).sum()
            #print("FP: ", FP.item())
            precisions = TP / (TP + FP)
            FN = ((y == 1) & (y_pred != 1)).sum()
            #print("FN: ", FN.item())
            recalls = TP / (TP + FN)
        elif y.task=="sum":
            accs = (y_pred == y).float().mean(dim=0)
            precisions = []
            recalls = []
            for i in range(3):
                TP = ((y == i) & (y_pred == i)).sum()
                FP = ((y != i) & (y_pred == i)).sum()
                precisions.append(TP / (TP + FP))
                FN = ((y == i) & (y_pred != i)).sum()
                recalls.append(TP / (TP + FN))
    return accs, precisions, recalls

def print_performances(accs, precisions, recalls, task, environment="train"):
    if task=="all":
        print(f"  {environment}:  Accuracy=[Y2F: {accs[0]:.3f}, B2F: {accs[1]:.3f}], Precision=[Y2F {precisions[0]:.3f}, B2F {precisions[1]:.3f}], Recall=[Y2F {recalls[0]:.3f}, B2F {recalls[1]:.3f}]")
    elif task in ["yellow", "blue", "or"]:
        print(f"  {environment}:  Accuracy={accs:.3f}, Precision={precisions:.3f}, Recall={recalls:.3f}")
    elif task=="sum":
        print(f"  {environment}:  Accuracy={accs:.3f}, Precision=[0: {precisions[0]:.3f}; 1: {precisions[1]:.3f}; 2: {precisions[2]:.3f}], Recall=[0: {recalls[0]:.3f}; 1: {recalls[1]:.3f}; 2: {recalls[2]:.3f}]")