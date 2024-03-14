import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import MLP


def train_model(X, y, test_size=0.2, random_state=42, batch_size=1024, num_epochs=20, lr=0.0001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=random_state,
                                                      shuffle=False)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    input_size = X.shape[1]
    hidden_size = 100
    output_size = y.shape[1] #int(y.max())+1

    model = MLP(input_size, hidden_size, output_size).to(device)
    BCE = torch.nn.BCELoss()
    CE = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch.to(device)
            y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = BCE(y_pred, y_batch) + CE(y_pred.sum(dim=1), y_batch.sum(dim=1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if verbose: print(f"  Train: Loss={train_loss / len(train_loader):.3f}")
        
        train_accs, train_f1scores = evaluate_model(model, X_train, y_train)
        if verbose: print(f"  Train:  Accuracy=[Y2F: {train_accs[0]:.3f}, B2F: {train_accs[1]:.3f}, sum: {train_accs[2]:.3f}], F1 Score=[Y2F {train_f1scores[0]:.3f}, B2F {train_f1scores[1]:.3f}]")
        val_accs, val_f1scores = evaluate_model(model, X_val, y_val)
        if verbose: print(f"  Val:    Accuracy=[Y2F: {val_accs[0]:.3f}, B2F: {val_accs[1]:.3f}, sum: {val_accs[2]:.3f}], F1 Score=[Y2F {val_f1scores[0]:.3f}, B2F {val_f1scores[1]:.3f}]")

    return model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model.pred(X)
        accs = [acc.item() for acc in (y_pred == y).float().mean(dim=0)]
        accs.append((y_pred.sum(dim=1) == y.sum(dim=1)).float().mean(dim=0))
        f1scores = [f1_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
    return accs, f1scores