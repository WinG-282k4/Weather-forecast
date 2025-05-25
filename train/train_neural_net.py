import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from settings import TARGET, FEATURES, BASE_DIR
from utils import save_result

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class WeatherClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim  # Đầu ra của lớp trước là đầu vào của lớp tiếp theo
            hidden_dim //= 2  # Giảm kích thước lớp ẩn (512 -> 256 -> 128 -> ...)

            # Lớp đầu ra
        layers.append(nn.Linear(input_dim, num_classes))

        # Tạo mạng từ danh sách các lớp
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)

# Đọc dữ liệu
df = pd.read_csv("/Clean_data/clean_data.csv")
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df[TARGET])

X = df[FEATURES].values
y = df['weather_encoded'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_dataset = WeatherDataset(X_train, y_train)
val_dataset = WeatherDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

input_dim = X.shape[1]
num_classes = len(le.classes_)
model = WeatherClassifier(input_dim, num_classes, hidden_dim=512, num_hidden_layers=3, dropout_rate=0.5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === EARLY STOPPING CONFIG ===
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Đánh giá trên validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Load mô hình tốt nhất
model.load_state_dict(best_model_state)
model.eval()

# Tính train_report
train_preds, train_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(y_batch.cpu().numpy())

train_report = classification_report(train_labels, train_preds, target_names=le.classes_)

# Tính val_report
val_preds, val_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(y_batch.cpu().numpy())

val_report = classification_report(val_labels, val_preds, target_names=le.classes_)

# Lưu kết quả
save_path = os.path.join(BASE_DIR, "model", "neural_net", "best_model")
save_result(save_path, train_report, val_report, scaler, model, None)
