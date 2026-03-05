import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# 0) Repro / Device
# =========================
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "| device:", device)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# =========================
# 1) Load data (LOCAL)
# =========================
csv_path = Path("../cicids_2017/Wednesday-workingHours.pcap_ISCX.csv")
if not csv_path.exists():
    raise FileNotFoundError(csv_path.resolve())

df = pd.read_csv(csv_path)
print("Raw shape:", df.shape)

# CIC-IDS2017 常見欄位前後有空白 → strip
df.columns = df.columns.str.strip()
assert "Label" in df.columns, "找不到 Label 欄位"
df["Label"] = df["Label"].astype(str).str.strip()

# =========================
# 2) Clean: duplicates, inf, missing
# =========================
before = len(df)
df = df.drop_duplicates()
print(f"Drop duplicates: {before} -> {len(df)}")

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

nan_before = df[numeric_cols].isna().sum().sum()
print("NaN before:", int(nan_before))

# 不建議 dropna（會亂刪類別）→ 用 median 補值
med = df[numeric_cols].median(numeric_only=True)
df[numeric_cols] = df[numeric_cols].fillna(med)

nan_after = df[numeric_cols].isna().sum().sum()
print("NaN after:", int(nan_after))

print(df["Label"].value_counts().head(10))

# =========================
# 3) Build X, y (multiclass)
# =========================
drop_cols = [c for c in ["Flow ID","Source IP","Source Port","Destination IP","Destination Port","Timestamp"] if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)

feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols].to_numpy(dtype=np.float32)

label_names = sorted(df["Label"].unique().tolist())
label2id = {name:i for i, name in enumerate(label_names)}
y = df["Label"].map(label2id).to_numpy(dtype=np.int64)

num_classes = len(label_names)
benign_id = label2id.get("BENIGN", None)
print("num_classes:", num_classes, "| BENIGN id:", benign_id)

# =========================
# 4) Split 70/10/20 + scaler (no leakage)
# =========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
# val=10% total => 0.10/0.80=0.125 of trainval
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.125, random_state=SEED, stratify=y_trainval
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# 1D CNN/序列模型：PyTorch Conv1d/LSTM 需要 (N, C, L) 或 (N, T, F)
X_train_cnn = X_train[:, None, :]  # (N,1,F)
X_val_cnn   = X_val[:, None, :]
X_test_cnn  = X_test[:, None, :]

X_train_seq = X_train[:, None, :]  # (N,1,F) 當作 1 timestep
X_val_seq   = X_val[:, None, :]
X_test_seq  = X_test[:, None, :]

# =========================
# 5) Dataset / Loader
# =========================
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

batch_size = 512
train_loader_cnn = DataLoader(NumpyDataset(X_train_cnn, y_train), batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
val_loader_cnn   = DataLoader(NumpyDataset(X_val_cnn, y_val),     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader_cnn  = DataLoader(NumpyDataset(X_test_cnn, y_test),   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

train_loader_seq = DataLoader(NumpyDataset(X_train_seq, y_train), batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
val_loader_seq   = DataLoader(NumpyDataset(X_val_seq, y_val),     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader_seq  = DataLoader(NumpyDataset(X_test_seq, y_test),   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# =========================
# 6) Models
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):  # x: (B,F) or (B,1,F)
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):  # x: (B,1,F)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Global average pool over length
        x = x.mean(dim=-1)  # (B,128)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, bidir=False, num_classes=2, dropout=0.3):
        super().__init__()
        self.bidir = bidir
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden1, batch_first=True, bidirectional=bidir)
        out1 = hidden1 * (2 if bidir else 1)
        self.lstm2 = nn.LSTM(input_size=out1, hidden_size=hidden2, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden2, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):  # x: (B,T,F)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last timestep
        return self.fc(x)

class GRUClassifier(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.gru1 = nn.GRU(input_size=in_dim, hidden_size=hidden1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden2, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        return self.fc(x)

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):  # x: (B,1,F)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B,128,L')
        x = x.permute(0, 2, 1)  # (B,L',128) -> 當成 time steps
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        return self.fc(x)

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64*2, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B,128,L')
        x = x.permute(0, 2, 1)  # (B,L',128)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.fc(x)

# =========================
# 7) Train / Eval utils
# =========================
def train_model(model, train_loader, val_loader, max_epochs=20, lr=1e-3, patience=5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val = 1e18
    best_state = None
    bad = 0

    for ep in range(1, max_epochs+1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * yb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += yb.size(0)

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += loss.item() * yb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total += yb.size(0)

        tr_loss /= max(1, tr_total)
        va_loss /= max(1, va_total)
        tr_acc = tr_correct / max(1, tr_total)
        va_acc = va_correct / max(1, va_total)

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def eval_model(model, test_loader):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(yb.numpy())
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"[Multiclass] ACC={acc:.4f} macroF1={macro:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("CM shape:", confusion_matrix(y_true, y_pred).shape)

    # binary projection for FPR/DR
    if benign_id is not None:
        y_true_bin = (y_true != benign_id).astype(np.int64)
        p_attack = 1.0 - probs[:, benign_id]
        thr = 0.5
        y_pred_bin = (p_attack >= thr).astype(np.int64)
        tn = int(((y_true_bin==0) & (y_pred_bin==0)).sum())
        fp = int(((y_true_bin==0) & (y_pred_bin==1)).sum())
        fn = int(((y_true_bin==1) & (y_pred_bin==0)).sum())
        tp = int(((y_true_bin==1) & (y_pred_bin==1)).sum())
        fpr = fp / (fp + tn + 1e-12)
        dr  = tp / (tp + fn + 1e-12)
        print(f"[Binary proj @0.5] FPR={fpr:.4f} DR/TPR={dr:.4f}  tn fp fn tp={tn} {fp} {fn} {tp}")

# =========================
# 8) Choose model & run
# =========================
in_dim = X_train.shape[1]

# 你要跑哪個模型就換這行：
model = CNN1D(num_classes=num_classes, dropout=0.3)
# model = MLP(in_dim=in_dim, num_classes=num_classes)
# model = LSTMClassifier(in_dim=in_dim, bidir=False, num_classes=num_classes)
# model = GRUClassifier(in_dim=in_dim, num_classes=num_classes)
# model = CNN_LSTM(num_classes=num_classes)
# model = CNN_BiLSTM(num_classes=num_classes)
# model = LSTMClassifier(in_dim=in_dim, bidir=True, num_classes=num_classes)  # BiLSTM 版本（不含 CNN）

# 訓練/驗證 loader 要對應 input shape：
# - CNN 系列用 *_cnn loader
# - LSTM/GRU 系列用 *_seq loader
if isinstance(model, (CNN1D, CNN_LSTM, CNN_BiLSTM)):
    trained = train_model(model, train_loader_cnn, val_loader_cnn, max_epochs=20, lr=1e-3, patience=5)
    eval_model(trained, test_loader_cnn)
elif isinstance(model, MLP):
    # MLP 用 (N,F)，這裡直接把 seq/cnn 的 X squeeze
    train_loader_mlp = DataLoader(NumpyDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader_mlp   = DataLoader(NumpyDataset(X_val, y_val),     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader_mlp  = DataLoader(NumpyDataset(X_test, y_test),   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    trained = train_model(model, train_loader_mlp, val_loader_mlp, max_epochs=20, lr=1e-3, patience=5)
    eval_model(trained, test_loader_mlp)
else:
    trained = train_model(model, train_loader_seq, val_loader_seq, max_epochs=20, lr=1e-3, patience=5)
    eval_model(trained, test_loader_seq)


import numpy as np
import torch
import pandas as pd
from pathlib import Path

@torch.no_grad()
def predict_probs(model, X_np, device, batch_size=2048):
    model.eval()
    probs_all = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.from_numpy(X_np[i:i+batch_size]).float().to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
    probs = np.concatenate(probs_all, axis=0)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

# ========= A) 預測 Wednesday test set =========
# 你已經有 X_test_cnn（形狀 (N,1,F)）與 y_test
probs_wed_test, yhat_wed_test = predict_probs(trained, X_test_cnn, device)

print("Wed test preds:", yhat_wed_test.shape, "probs:", probs_wed_test.shape)

# 若你想把預測存成檔案（含真實標籤）
wed_out = pd.DataFrame({
    "y_true": y_test,
    "y_pred": yhat_wed_test,
    "p_max": probs_wed_test.max(axis=1),
})
wed_out.to_csv("pred_wednesday_test.csv", index=False)
print("Saved: pred_wednesday_test.csv")


# ========= B) 預測完整 Friday（以 Friday-DDoS 為例） =========
fri_path = Path("../cicids_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df_fri = pd.read_csv(fri_path)

# 1) 欄位 strip（CICIDS 常見空白）
df_fri.columns = df_fri.columns.str.strip()
if "Label" in df_fri.columns:
    df_fri["Label"] = df_fri["Label"].astype(str).str.strip()

# 2) 同樣清理：drop dup, inf->nan, median impute
df_fri = df_fri.drop_duplicates()
num_cols = df_fri.select_dtypes(include=[np.number]).columns
df_fri[num_cols] = df_fri[num_cols].replace([np.inf, -np.inf], np.nan)
df_fri[num_cols] = df_fri[num_cols].fillna(df_fri[num_cols].median(numeric_only=True))

# 3) drop 非特徵欄（跟 Wednesday 一致）
drop_cols = [c for c in ["Flow ID","Source IP","Source Port","Destination IP","Destination Port","Timestamp"] if c in df_fri.columns]
df_fri = df_fri.drop(columns=drop_cols, errors="ignore")

# 4) ★最重要：Friday 必須用「Wednesday 的 feature_cols」對齊欄位
# feature_cols 是你 Wednesday 那段建立 X 時用的欄位清單（不含 Label）
missing = [c for c in feature_cols if c not in df_fri.columns]
if missing:
    raise ValueError(f"Friday 缺少特徵欄位（無法對齊）：{missing[:10]} ... total {len(missing)}")

X_fri = df_fri[feature_cols].to_numpy(dtype=np.float32)

# 5) ★用 Wednesday train fit 出來的 scaler 來 transform（不要重新 fit）
X_fri_s = scaler.transform(X_fri).astype(np.float32)

# 6) reshape 給 CNN： (N,1,F)
X_fri_cnn = X_fri_s[:, None, :]

# 7) predict
probs_fri, yhat_fri = predict_probs(trained, X_fri_cnn, device)

# === 直接計算 Friday binary metrics（不用 csv） ===
# 先取得 Friday 的真實 label（你已經有 df_fri["Label"]）
y_fri_true = df_fri["Label"].values
y_fri_true_bin = (y_fri_true != "BENIGN").astype(np.int64)  # 0=benign, 1=attack

# 預測機率 p_attack
p_attack_fri = 1.0 - probs_fri[:, benign_id]
y_pred_bin_fri = (p_attack_fri >= 0.5).astype(np.int64)

tn = ((y_fri_true_bin == 0) & (y_pred_bin_fri == 0)).sum()
fp = ((y_fri_true_bin == 0) & (y_pred_bin_fri == 1)).sum()
fn = ((y_fri_true_bin == 1) & (y_pred_bin_fri == 0)).sum()
tp = ((y_fri_true_bin == 1) & (y_pred_bin_fri == 1)).sum()

acc_fri = (tp + tn) / (tp + tn + fp + fn + 1e-12)
dr_fri  = tp / (tp + fn + 1e-12)
fpr_fri = fp / (fp + tn + 1e-12)

print("\n=== Friday (no TL, zero-shot) Binary Metrics (direct calc) ===")
print(f"ACC: {acc_fri:.4f}")
print(f"DR/TPR: {dr_fri:.4f}")
print(f"FPR: {fpr_fri:.4f}")
print(f"tn fp fn tp = {tn} {fp} {fn} {tp}")
print(f"Total samples: {len(y_fri_true_bin)}")

print("Friday preds:", yhat_fri.shape, "probs:", probs_fri.shape)

# 存檔：可把每筆預測類別 + confidence 存起來
fri_out = pd.DataFrame({
    "y_pred": yhat_fri,
    "p_max": probs_fri.max(axis=1),
})

# 若 Friday 有 Label 欄位，且你想一起存（但注意：Friday label space 不一定跟 Wednesday 一樣）
if "Label" in df_fri.columns:
    fri_out["label_raw"] = df_fri["Label"].values

fri_out.to_csv("pred_friday_full.csv", index=False)
print("Saved: pred_friday_full.csv")

# ===== Binary labels =====
# 假設你還保留原本 df（或至少有 label_raw 的陣列）
# 若你沒有 df 了，也可以用原本 y + benign_id 轉二分類：
y_bin = (y != benign_id).astype(np.int64)   # 0=BENIGN, 1=ATTACK

# 重新切分（建議重新切，確保二分類比例一致）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y_bin, test_size=0.20, random_state=SEED, stratify=y_bin
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.125, random_state=SEED, stratify=y_trainval
)

# scaler 只 fit train（如果你上面已經做過，可以直接沿用那份 scaler，不必重做）
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

X_train_cnn = X_train_s[:, None, :]
X_val_cnn   = X_val_s[:, None, :]
X_test_cnn  = X_test_s[:, None, :]

# ===== 換成二分類模型：num_classes=2 =====
num_classes_bin = 2
model_bin = CNN1D(num_classes=num_classes_bin, dropout=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_bin.parameters(), lr=1e-3)

# 你原本的 train loop 如果寫死用 model/optimizer/criterion
# 建議你把 train loop 包成函數，或暫時把裡面 model 改成 model_bin

# 訓練完後評估
probs_bin, yhat_bin = predict_probs(model_bin, X_test_cnn, device)
# 注意：predict_probs 會輸出 2 類的 probs

import pandas as pd
import numpy as np

BENIGN_ID = 0  # 你已確認 label2id['BENIGN']=0

def bin_metrics_from_csv(csv_path: str, benign_id: int = 0):
    df = pd.read_csv(csv_path)

    # ===== 真值 =====
    if "y_true" in df.columns:
        y_true = df["y_true"].to_numpy()
        y_true_bin = (y_true != benign_id).astype(np.int64)
    elif "label_raw" in df.columns:
        y_true_bin = (df["label_raw"].astype(str).str.strip().to_numpy() != "BENIGN").astype(np.int64)
    else:
        raise ValueError(f"{csv_path} 缺少 y_true 或 label_raw，無法算 ACC/DR/FPR")

    # ===== 預測 =====
    y_pred = df["y_pred"].to_numpy()
    y_pred_bin = (y_pred != benign_id).astype(np.int64)

    tn = int(((y_true_bin==0) & (y_pred_bin==0)).sum())
    fp = int(((y_true_bin==0) & (y_pred_bin==1)).sum())
    fn = int(((y_true_bin==1) & (y_pred_bin==0)).sum())
    tp = int(((y_true_bin==1) & (y_pred_bin==1)).sum())

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    dr  = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)

    return {"ACC": acc, "DR": dr, "FPR": fpr, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "N": int(len(df))}

BENIGN_ID = 0

for f in ["pred_wednesday_test.csv", "pred_friday_full.csv"]:
    print(f, bin_metrics_from_csv(f, benign_id=BENIGN_ID))