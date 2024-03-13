import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import MLP


def train_model(X, y, test_size=0.2, random_state=42, batch_size=1024, num_epochs=10, lr=0.001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=random_state)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    hidden_size = 100
    output_size = int(y.max())+1

    model = MLP(input_size, hidden_size, output_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
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
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if verbose: print(f"  Train: Loss={train_loss / len(train_loader)}")
        
        train_acc, train_f1score = evaluate_model(model, train_loader)
        if verbose: print(f"  Train: Accuracy={train_acc}, F1 Score={train_f1score}")
        val_acc, val_f1score = evaluate_model(model, val_loader)
        if verbose: print(f"  Val:   Accuracy={val_acc}, F1 Score={val_f1score}")

    return model

def evaluate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        acc = sum((model.pred(X_batch) == y_batch).sum().item() for X_batch, y_batch in data_loader) /len(data_loader)
        f1score = sum(f1_score(y_batch, model.pred(X_batch), average="macro") for X_batch, y_batch in data_loader) /len(data_loader)
    return acc, f1score