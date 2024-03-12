from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import tqdm
import torch

from data import get_data_sl

X, y = get_data_sl(environment="train", model_name="dino", outcome="sum")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = nn.Sequential(
    nn.Linear(X.shape[1], 100),
    nn.ReLU(),
    nn.Linear(100, 3)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0
    train_acc = 0
    for X_batch, y_batch in tqdm(train_loader):
        X_batch.to(device)
        y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y_batch).sum().item()
    print(f"  Train: Loss={train_loss / len(train_loader)}, Accuracy={train_acc / len(X_train)}")
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(X_batch), y_batch) for X_batch, y_batch in val_loader) / len(val_loader)
        val_acc = sum((model(X_batch).argmax(dim=1) == y_batch).sum().item() for X_batch, y_batch in val_loader) / len(X_val)
    print(f"  Val:   Loss={val_loss}, Accuracy={val_acc}")