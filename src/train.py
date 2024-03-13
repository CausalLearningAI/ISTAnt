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
            loss = criterion(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if verbose: print(f"  Train: Loss={train_loss / len(train_loader):.3f}")
        
        train_acc, train_f1score = evaluate_model(model, X_train, y_train)
        if verbose: print(f"  Train: Accuracy={train_acc:.3f}, F1 Score={train_f1score:.3f}")
        val_acc, val_f1score = evaluate_model(model, X_val, y_val)
        if verbose: print(f"  Val:   Accuracy={val_acc:.3f}, F1 Score={val_f1score:.3f}")

    return model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        acc = (model.pred(X) == y).float().mean().item()
        f1score = f1_score(y, model.pred(X), average="macro")
    return acc, f1score