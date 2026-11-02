"""
Progetto CSL – PneumoniaMNIST
Confronto tra SVM (PCA/HOG) e CNN (SGD vs Adam)
"""

# ============================================================
# 1. IMPORT E CONFIGURAZIONE
# ============================================================

import time

import os
import json
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

import medmnist
from medmnist import PneumoniaMNIST

from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)

from skimage.feature import hog


# IMPOSTAZIONE SEED PER RIPRODUCIBILITÀ
# import random

# SEED = 42

# Seed Python "puro"
# random.seed(SEED)

# Seed NumPy
# np.random.seed(SEED)

# Seed PyTorch
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# Per garantire consistenza delle operazioni CUDA (se hai GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Impostazioni generali
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results_compact"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. CARICAMENTO DATASET PNEUMONIAMNIST
#    + CONVERSIONE IN NUMPY PER SVM
# ============================================================

def get_pneumonia_loaders(batch_size: int = 64, download: bool = True
                          ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Ritorna DataLoader per train/val/test di PneumoniaMNIST.
    """
    data_flag = 'pneumoniamnist'
    _ = medmnist.INFO[data_flag]

    # Converte le immagini (PIL) in tensori torch (C x H x W) in [0,1]
    transform = transforms.ToTensor()

    train_ds = PneumoniaMNIST(split='train', transform=transform, download=download)
    val_ds   = PneumoniaMNIST(split='val',   transform=transform, download=download)
    test_ds  = PneumoniaMNIST(split='test',  transform=transform, download=download)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def loaders_to_numpy(train_loader, val_loader, test_loader):
    """
    Converte i DataLoader in array numpy (N, H, W) e label (N,)
    per l'uso con scikit-learn/SVM.
    """
    def loader_to_np(loader):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy())  # (B, 1, 28, 28)
            ys.append(y.numpy())  # (B, 1)
        xs = np.concatenate(xs, axis=0)      # (N, 1, 28, 28)
        ys = np.concatenate(ys, axis=0)      # (N, 1)
        xs = xs[:, 0, :, :]                  # (N, 28, 28) rimuovo canale
        ys = ys.reshape(-1)                  # (N,)
        return xs, ys

    X_train, y_train = loader_to_np(train_loader)
    X_val,   y_val   = loader_to_np(val_loader)
    X_test,  y_test  = loader_to_np(test_loader)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 3. PIPELINE SVM + PCA
# ============================================================

def compute_pca_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    variance_ratio: float = 0.95
):
    """
    Flatten -> StandardScaler -> PCA (95% varianza).
    """
    def flatten(X):
        return X.reshape(X.shape[0], -1)

    X_train_f = flatten(X_train)
    X_val_f   = flatten(X_val)
    X_test_f  = flatten(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_val_s   = scaler.transform(X_val_f)
    X_test_s  = scaler.transform(X_test_f)

    pca = PCA(n_components=variance_ratio, svd_solver='full')
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p   = pca.transform(X_val_s)
    X_test_p  = pca.transform(X_test_s)

    return X_train_p, X_val_p, X_test_p


def train_eval_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_type: str = "pca",
) -> Dict:
    """
    Allena una SVM RBF con GridSearchCV (roc_auc) e valuta su test.
    feature_type: solo per etichettare i risultati (pca / hog).
    """
    svm = SVC(kernel="rbf", probability=True)

    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [1e-3, 1e-2, 1e-1],
    }

    grid = GridSearchCV(
        svm,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # Validation
    val_proba = best.predict_proba(X_val)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)

    val_metrics = {
        "val_auc":   float(roc_auc_score(y_val, val_proba)),
        "val_acc":   float(accuracy_score(y_val, val_pred)),
        "val_f1":    float(f1_score(y_val, val_pred)),
        "val_brier": float(brier_score_loss(y_val, val_proba)),
        "best_params": grid.best_params_,
    }

    # Test
    test_proba = best.predict_proba(X_test)[:, 1]
    test_pred  = (test_proba >= 0.5).astype(int)

    test_metrics = {
        "test_auc":   float(roc_auc_score(y_test, test_proba)),
        "test_acc":   float(accuracy_score(y_test, test_pred)),
        "test_f1":    float(f1_score(y_test, test_pred)),
        "test_brier": float(brier_score_loss(y_test, test_proba)),
    }

    results = {
        "feature_type": feature_type,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }

    print(f"\n=== SVM ({feature_type}) ===")
    print(json.dumps(results, indent=4))

    # salvo su file
    out_path = os.path.join(RESULTS_DIR, f"svm_{feature_type}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


# ============================================================
# 4. PIPELINE SVM + HOG
# ============================================================

def compute_hog_features(
    X: np.ndarray,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    orientations: int = 9
) -> np.ndarray:
    feats = []
    for img in X:
        f = hog(
            img,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
            visualize=False,
            feature_vector=True,
        )
        feats.append(f)
    return np.array(feats)


# ============================================================
# 5. DEFINIZIONE CNN COMPATTA
# ============================================================

class SmallPneumoniaCNN(nn.Module):
    """
    Architettura del progetto:
    Conv(1->16,3x3,pad1) -> BN -> ReLU -> MaxPool(2x2)
    Conv(16->32,3x3,pad1)-> BN -> ReLU -> MaxPool(2x2)
    Conv(32->64,3x3,pad1)-> BN -> ReLU
    GAP -> Dropout(0.3) -> FC(64->1)
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)
        self.fc      = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,16,14,14)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,32,7,7)
        x = F.relu(self.bn3(self.conv3(x)))             # (B,64,7,7)
        x = x.mean(dim=[2, 3])                          # GAP -> (B,64)
        x = self.dropout(x)
        x = self.fc(x)                                  # (B,1)
        return x.squeeze(1)                             # (B,)


# ============================================================
# 6. TRAINING CNN (SGD O ADAM) + METRICHE
# ============================================================

def binary_classification_metrics(y_true, y_scores, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "auc":   float(roc_auc_score(y_true, y_scores)),
        "acc":   float(accuracy_score(y_true, y_pred)),
        "f1":    float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_scores)),
    }


def collect_probs_and_labels(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.view(-1).to(device).float()
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_labels, all_probs


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.view(-1).to(device).float()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def train_cnn(
    optimizer_name: str = "sgd",
    epochs: int = 30,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    lr_sgd: float = 0.05,
    lr_adam: float = 1e-3,
    momentum: float = 0.9,
    patience: int = 5,
) -> Dict:
    """
    Allena la CNN con SGD+Momentum o Adam e restituisce storia e metriche test.
    """
    train_loader, val_loader, test_loader = get_pneumonia_loaders(batch_size=batch_size)
    model = SmallPneumoniaCNN(dropout_p=0.3).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr_sgd,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr_adam,
            weight_decay=weight_decay,
        )
        scheduler = None

    best_val_auc = -1.0
    best_state = None
    history = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        y_val, p_val = collect_probs_and_labels(model, val_loader, DEVICE)
        metrics_val = binary_classification_metrics(y_val, p_val)

        if scheduler is not None:
            scheduler.step()

        if metrics_val["auc"] > best_val_auc:
            best_val_auc = metrics_val["auc"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": metrics_val,
        })

        print(
            f"[{optimizer_name.upper()}] Epoch {epoch:03d} | "
            f"loss: {train_loss:.4f} | val_auc: {metrics_val['auc']:.4f} | "
            f"val_acc: {metrics_val['acc']:.4f}"
        )

        if patience_counter >= patience:
            print(f"[{optimizer_name.upper()}] Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = collect_probs_and_labels(model, test_loader, DEVICE)
    metrics_test = binary_classification_metrics(y_test, p_test)

    results = {
        "optimizer": optimizer_name,
        "history": history,
        "best_val_auc": best_val_auc,
        "test_metrics": metrics_test,
    }

    # salvo
    out_json = os.path.join(RESULTS_DIR, f"cnn_{optimizer_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    out_weights = os.path.join(RESULTS_DIR, f"cnn_{optimizer_name}_best.pt")
    if best_state is not None:
        torch.save(best_state, out_weights)

    print(f"\n=== CNN ({optimizer_name}) test metrics ===")
    print(json.dumps(metrics_test, indent=4))

    return results


# ============================================================
# 7. FUNZIONI DI SUPPORTO PER PLOT (LEARNING CURVES)
# ============================================================

def plot_learning_curves(history: Dict, title_prefix: str = "CNN"):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_auc = [h["val_metrics"]["auc"] for h in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Training loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, val_auc, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{title_prefix} - Validation AUC")
    plt.legend()
    plt.show()


# ============================================================
# 8. MAIN: LANCIO ESPERIMENTI
# ============================================================

def main():
    timings = {}  # per salvare i tempi di ogni metodo

    # Carico dati una sola volta per SVM
    train_loader, val_loader, test_loader = get_pneumonia_loaders(batch_size=128)
    X_train, y_train, X_val, y_val, X_test, y_test = loaders_to_numpy(
        train_loader, val_loader, test_loader
    )

    # ---- SVM + PCA ----
    t0 = time.time()
    X_train_p, X_val_p, X_test_p = compute_pca_features(X_train, X_val, X_test)
    svm_pca_results = train_eval_svm(
        X_train_p, y_train, X_val_p, y_val, X_test_p, y_test, feature_type="pca"
    )
    timings["svm_pca_sec"] = time.time() - t0

    # ---- SVM + HOG ----
    t0 = time.time()
    X_train_h = compute_hog_features(X_train)
    X_val_h   = compute_hog_features(X_val)
    X_test_h  = compute_hog_features(X_test)

    scaler_h = StandardScaler()
    X_train_hs = scaler_h.fit_transform(X_train_h)
    X_val_hs   = scaler_h.transform(X_val_h)
    X_test_hs  = scaler_h.transform(X_test_h)

    svm_hog_results = train_eval_svm(
        X_train_hs, y_train, X_val_hs, y_val, X_test_hs, y_test, feature_type="hog"
    )
    timings["svm_hog_sec"] = time.time() - t0


    # ---- CNN con SGD ----
    # t0 = time.time()
    # cnn_sgd_results = train_cnn(optimizer_name="sgd")
    # timings["cnn_sgd_sec"] = time.time() - t0

    # ---- CNN con SGD (esperimento 2) ----
    t0 = time.time()
    cnn_sgd_results = train_cnn(
       optimizer_name="sgd",
       lr_sgd=0.01,      # <— learning rate più basso
       epochs=40         # <— più epoche per compensare lr minore
    )
    timings["cnn_sgd_sec"] = time.time() - t0

    # ---- CNN con Adam ----
    t0 = time.time()
    cnn_adam_results = train_cnn(optimizer_name="adam")
    timings["cnn_adam_sec"] = time.time() - t0

    # ---- Salvataggio e stampa tempi di esecuzione ----
    print("\n=== Tempi di esecuzione (secondi) ===")
    for k, v in timings.items():
        print(f"{k}: {v:.2f} s")

    timings_path = os.path.join(RESULTS_DIR, "timings.json")
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=4)
    print(f"\nTempi salvati in {timings_path}")

    # Plot delle learning curves
    plot_learning_curves(cnn_sgd_results["history"], title_prefix="CNN SGD")
    plot_learning_curves(cnn_adam_results["history"], title_prefix="CNN Adam")


if __name__ == "__main__":
    main()
