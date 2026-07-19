"""
10_train_pipeline.py
====================
Comprehensive Taamify Training Pipeline (MPS-first).

Features:
- 9 architectures: MLP baseline, CNN, DeepCNN, ResNetMini, CRNN (GRU),
  CRNN-LSTM, BiLSTM, pretrained ResNet18 (transfer learning), SpecTransformer
- Full regularization: Dropout, BatchNorm, L2 (weight_decay), SpecAugment, Early Stopping
- Complete metrics: Accuracy, Balanced Accuracy, MCC, F1, AUC-ROC (per-class + macro)
- Proper Train/Val/Test evaluation with ROC curves, confusion matrices, comparison chart
- Device auto-selection: MPS (Apple Silicon) -> CUDA -> CPU; AMP on CUDA only
- Penultimate-layer embedding export for the ta'am-similarity analysis

Usage:
    python 10_train_pipeline.py --data-dir prepared_data/random_top11
    python 10_train_pipeline.py --data-dir prepared_data/random_top11 --model all
    python 10_train_pipeline.py --data-dir prepared_data/loro_R2_top11 --model resnet18_tl
"""

import argparse
import json
import numpy as np
import copy
import os
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc,
    f1_score, precision_recall_fscore_support,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = PROJECT_DIR / "prepared_data" / "random_top11"
RESULTS_ROOT = PROJECT_DIR / "results"

# Set seeds (torch.manual_seed also seeds MPS)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ═══════════════════════════════════════════════════════════════════════════
#  SPECAUGMENT
# ═══════════════════════════════════════════════════════════════════════════

def spec_augment(x, freq_mask_param=15, time_mask_param=15,
                 num_freq_masks=2, num_time_masks=2):
    """Apply SpecAugment to a batch of spectrograms (B, 1, H, W)."""
    B, _, H, W = x.shape
    y = x.clone()
    for i in range(B):
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param + 1)
            f0 = np.random.randint(0, max(1, H - f))
            if f > 0:
                y[i, :, f0:f0 + f, :] = 0.0
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param + 1)
            t0 = np.random.randint(0, max(1, W - t))
            if t > 0:
                y[i, :, :, t0:t0 + t] = 0.0
    return y


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════

class CNN_Baseline(nn.Module):
    """Original 3-block CNN (baseline). ~1.9M params."""
    name = "CNN_Baseline"

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 10, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepCNN(nn.Module):
    """Deeper 5-block CNN with more capacity. ~3.5M params."""
    name = "DeepCNN"

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x87 -> 64x43
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            # Block 2: 64x43 -> 32x21
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            # Block 3: 32x21 -> 16x10
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            # Block 4: 16x10 -> 8x5
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        # After 4 MaxPool: 128->64->32->16->8 height, 87->43->21->10->5 width
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 5, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetMini(nn.Module):
    """Mini ResNet-style architecture for spectrograms. ~2.8M params."""
    name = "ResNetMini"

    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)    # 128x87 -> 64x44
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 64x44 -> 32x22
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 32x22 -> 16x11
        self.dropout = nn.Dropout2d(0.2)
        # Global average pooling: output size (1,1) divides any input, so it is
        # safe on MPS (adaptive pool to (4,4) crashes: 16x11 not divisible)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dropout(self.layer1(x))
        x = self.dropout(self.layer2(x))
        x = self.dropout(self.layer3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CRNN(nn.Module):
    """CNN + Bidirectional GRU for sequential pattern capture. ~2.1M params."""
    name = "CRNN"

    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # only reduce height, keep time resolution
            nn.Dropout2d(0.2),
        )
        # After CNN: H = 128/2/2/2 = 16, W = 87/2/2 = 21
        # Reshape to (batch, time=21, features=128*16=2048) for RNN
        self.rnn = nn.GRU(
            input_size=128 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),  # 256 = 128 * 2 (bidirectional)
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        # Reshape: (B, C, H, W) -> (B, W, C*H) — time steps along width
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        # RNN
        rnn_out, _ = self.rnn(x)
        # Use last hidden state (both directions concatenated)
        x = rnn_out[:, -1, :]
        return self.classifier(x)


class MLP(nn.Module):
    """Flat fully-connected baseline — the scientific control showing what the
    convolutional/recurrent structure actually buys. ~5.8M params."""
    name = "MLP"

    def __init__(self, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 87, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class BiLSTM(nn.Module):
    """Pure recurrent model: the spectrogram as a sequence of 87 time frames
    (128 mel features each) -> 2-layer BiLSTM -> mean+max pooling. ~0.8M params."""
    name = "BiLSTM"

    def __init__(self, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),  # 512 = 2*256 (mean||max)
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # (B, 1, 128, 87) -> (B, 87, 128): time steps along width
        x = x.squeeze(1).permute(0, 2, 1)
        out, _ = self.lstm(x)                       # (B, 87, 256)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.classifier(pooled)


class CRNN_LSTM(nn.Module):
    """CRNN with LSTM instead of GRU — direct comparison against CRNN. ~2.4M params."""
    name = "CRNN_LSTM"

    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
        )
        self.rnn = nn.LSTM(
            input_size=128 * 16, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        rnn_out, _ = self.rnn(x)
        return self.classifier(rnn_out[:, -1, :])


class ResNet18TL(nn.Module):
    """ImageNet-pretrained ResNet18, input adapted 3->1 channels (pretrained
    conv1 weights summed over RGB), fine-tuned end-to-end. ~11.2M params."""
    name = "ResNet18_TL"

    def __init__(self, num_classes=5):
        super().__init__()
        from torchvision.models import resnet18
        try:
            from torchvision.models import ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:  # offline / download failure
            print(f"  WARNING: pretrained weights unavailable ({e}), training from scratch")
            backbone = resnet18(weights=None)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class VGG16TL(nn.Module):
    """ImageNet-pretrained VGG-16 conv backbone (input adapted 3->1 channels),
    compact classifier head. The classic 25088-dim FC head is replaced — our
    inputs are 128x87, not 224x224 — leaving ~15.5M params. Included as the
    'many parameters, no skip connections' contrast to ResNet18_TL.
    avgpool is (2,2) because MPS adaptive pooling needs divisible sizes."""
    name = "VGG16_TL"

    def __init__(self, num_classes=5):
        super().__init__()
        from torchvision.models import vgg16
        try:
            from torchvision.models import VGG16_Weights
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception as e:  # offline / download failure
            print(f"  WARNING: pretrained weights unavailable ({e}), training from scratch")
            backbone = vgg16(weights=None)
        old_conv = backbone.features[0]
        new_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            new_conv.bias.copy_(old_conv.bias)
        backbone.features[0] = new_conv
        self.features = backbone.features           # (B,512,4,2) for 128x87
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # divisible -> MPS-safe
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 512), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


class SpecTransformer(nn.Module):
    """Small conv frontend -> Transformer encoder over time steps with a CLS
    token -> classification head. ~1.6M params."""
    name = "SpecTransformer"

    def __init__(self, num_classes=5, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),                    # 128x87 -> 64x43
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                  # 64x43 -> 32x43
            nn.Dropout2d(0.15),
        )
        self.proj = nn.Linear(64 * 32, d_model)    # per-time-step features -> d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 44, d_model))  # 43 steps + CLS
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.2, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.frontend(x)                       # (B, 64, 32, 43)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        x = self.proj(x)                           # (B, 43, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, :x.size(1) + 1]
        x = self.encoder(x)
        x = self.norm(x[:, 0])                     # CLS token
        return self.classifier(x)


MODEL_REGISTRY = {
    'mlp': MLP,
    'cnn': CNN_Baseline,
    'deep_cnn': DeepCNN,
    'resnet': ResNetMini,
    'crnn': CRNN,
    'crnn_lstm': CRNN_LSTM,
    'bilstm': BiLSTM,
    'resnet18_tl': ResNet18TL,
    'vgg16_tl': VGG16TL,
    'transformer': SpecTransformer,
}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, epoch, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def load_data(data_dir: Path, device_type: str):
    """Load prepared train/val/test splits."""
    print("\n[DATA] Loading prepared data...")

    train_data = np.load(data_dir / 'train.npz')
    val_data = np.load(data_dir / 'val.npz')
    test_data = np.load(data_dir / 'test.npz')

    with open(data_dir / 'metadata.json', 'r') as f:
        meta = json.load(f)

    X_train = torch.FloatTensor(train_data['X'])
    y_train = torch.LongTensor(train_data['y'])
    X_val = torch.FloatTensor(val_data['X'])
    y_val = torch.LongTensor(val_data['y'])
    X_test = torch.FloatTensor(test_data['X'])
    y_test = torch.LongTensor(test_data['y'])

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {meta['class_names']}")

    return (X_train, y_train, X_val, y_val, X_test, y_test, meta)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs=50, patience=10, use_specaugment=True,
                use_amp=True):
    """Train a model with full regularization pipeline."""
    early_stopping = EarlyStopping(patience=patience)
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_bal_acc': [], 'val_bal_acc': [],
        'lr': [],
    }

    print(f"\n  Training for up to {num_epochs} epochs (patience={patience})...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # ── Training ──
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if use_specaugment:
                inputs = spec_augment(inputs)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_labels)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        train_bal_acc = balanced_accuracy_score(train_labels, train_preds)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss /= len(val_labels_list)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels_list))
        val_bal_acc = balanced_accuracy_score(val_labels_list, val_preds)

        # Update LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_bal_acc'].append(train_bal_acc)
        history['val_bal_acc'].append(val_bal_acc)
        history['lr'].append(current_lr)

        # Print progress
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
              f"BalAcc: {train_bal_acc:.4f}/{val_bal_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        # Early stopping
        early_stopping(val_loss, epoch + 1, model)
        if early_stopping.early_stop:
            print(f"\n  Early stopping at epoch {epoch+1}. "
                  f"Best at epoch {early_stopping.best_epoch}")
            break

    # Restore best model
    model.load_state_dict(early_stopping.best_model_state)
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.1f}s. Best epoch: {early_stopping.best_epoch}")

    history['best_epoch'] = early_stopping.best_epoch
    history['training_time'] = elapsed

    return model, history


def extract_embeddings(model, data_loader, device):
    """Penultimate-layer activations (input to the final Linear) plus logits,
    for the ta'am-similarity analysis. Returns (embeddings, labels, logits)."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m  # registration order -> ends at the output layer

    captured, captured_logits = [], []

    def hook(_module, inputs, output):
        captured.append(inputs[0].detach().cpu())
        captured_logits.append(output.detach().cpu())

    handle = last_linear.register_forward_hook(hook)
    model.eval()
    labels = []
    with torch.no_grad():
        for inputs, y in data_loader:
            model(inputs.to(device))
            labels.append(y)
    handle.remove()
    return (torch.cat(captured).numpy(), torch.cat(labels).numpy(),
            torch.cat(captured_logits).numpy())


def evaluate_model(model, data_loader, device, class_names, split_name="Test"):
    """Evaluate model and compute all metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    num_classes = len(class_names)

    # Core metrics
    accuracy = np.mean(all_preds == all_labels)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # AUC-ROC (one-vs-rest)
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc_macro = auc_weighted = float('nan')

    # Per-class AUC
    per_class_auc = {}
    for i, name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        try:
            per_class_auc[name] = roc_auc_score(binary_labels, all_probs[:, i])
        except ValueError:
            per_class_auc[name] = float('nan')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'per_class_auc': per_class_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
    }

    print(f"\n  === {split_name} Results ===")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  MCC:               {mcc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  AUC-ROC (macro):   {auc_macro:.4f}")
    print(f"  AUC-ROC (weighted):{auc_weighted:.4f}")
    print(f"\n  Per-class AUC:")
    for name, auc_val in per_class_auc.items():
        print(f"    {name:25s}: {auc_val:.4f}")
    print(f"\n{report}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, model_name, output_dir):
    """Plot training curves: loss, accuracy, balanced accuracy, LR."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(history['train_loss']) + 1)
    best_epoch = history.get('best_epoch', len(epochs))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=3)
    ax.plot(epochs, history['val_loss'], 'r-o', label='Validation', markersize=3)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_title('Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=3)
    ax.plot(epochs, history['val_acc'], 'r-o', label='Validation', markersize=3)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Balanced Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['train_bal_acc'], 'b-o', label='Train', markersize=3)
    ax.plot(epochs, history['val_bal_acc'], 'r-o', label='Validation', markersize=3)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_title('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'g-o', markersize=3)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} - Training Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_training_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrices(metrics, class_names, model_name, output_dir):
    """Plot both raw count and normalized confusion matrices."""
    cm = metrics['confusion_matrix']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized (per row)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    fig.suptitle(f'{model_name} - Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(metrics, class_names, model_name, output_dir):
    """Plot ROC curves for each class (one-vs-rest) + macro average."""
    all_labels = metrics['all_labels']
    all_probs = metrics['all_probs']
    num_classes = len(class_names)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    # Per-class ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i, (name, color) in enumerate(zip(class_names, colors)):
        binary_labels = (all_labels == i).astype(int)
        try:
            fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {roc_auc_val:.3f})')
            # Interpolate for macro
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        except ValueError:
            pass

    # Macro-average ROC
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, 'k--', lw=3,
                label=f'Macro-average (AUC = {mean_auc:.3f})')

    # Diagonal
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1, alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_roc_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_summary(metrics, class_names, model_name, output_dir):
    """Plot a summary bar chart of key metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Accuracy', 'Balanced\nAccuracy', 'MCC', 'F1\n(macro)',
                    'F1\n(weighted)', 'AUC-ROC\n(macro)']
    metric_values = [
        metrics['accuracy'],
        metrics['balanced_accuracy'],
        metrics['mcc'],
        metrics['f1_macro'],
        metrics['f1_weighted'],
        metrics['auc_macro'],
    ]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4']
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.6, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} - Metrics Summary (Test Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1/len(class_names), color='red', linestyle='--', alpha=0.5,
               label=f'Random baseline ({1/len(class_names):.2f})')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_metrics_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_metrics_report(metrics, history, model_name, class_names, device, output_dir):
    """Save detailed text metrics report."""
    report_path = output_dir / f'{model_name}_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  TaamAI - {model_name} Training Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Training time: {history['training_time']:.1f}s\n")
        f.write(f"Epochs run: {len(history['train_loss'])}\n")
        f.write(f"Best epoch: {history['best_epoch']}\n\n")

        f.write("=" * 70 + "\n")
        f.write("  KEY METRICS (Test Set)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"  MCC:                {metrics['mcc']:.4f}\n")
        f.write(f"  F1 (macro):         {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}\n")
        f.write(f"  AUC-ROC (macro):    {metrics['auc_macro']:.4f}\n")
        f.write(f"  AUC-ROC (weighted): {metrics['auc_weighted']:.4f}\n\n")

        f.write("Per-class AUC-ROC:\n")
        for name, auc_val in metrics['per_class_auc'].items():
            f.write(f"  {name:25s}: {auc_val:.4f}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("  CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(metrics['classification_report'])
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("  CONFUSION MATRIX\n")
        f.write("=" * 70 + "\n\n")
        cm = metrics['confusion_matrix']
        header = f"{'':25s}" + "".join(f"{n:>12s}" for n in class_names)
        f.write(header + "\n")
        for i, name in enumerate(class_names):
            row = f"{name:25s}" + "".join(f"{cm[i, j]:12d}" for j in range(len(class_names)))
            f.write(row + "\n")

    print(f"  Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON ACROSS MODELS
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(all_results, output_dir):
    """Compare all trained models side by side."""
    if len(all_results) < 2:
        return

    model_names = list(all_results.keys())
    metric_keys = ['accuracy', 'balanced_accuracy', 'mcc', 'f1_macro', 'auc_macro']
    metric_labels = ['Accuracy', 'Balanced Acc', 'MCC', 'F1 (macro)', 'AUC-ROC']

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metric_labels))
    width = 0.8 / len(model_names)
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [all_results[name]['metrics'][k] for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Architecture Comparison (Test Set)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Save comparison table
    with open(output_dir / 'model_comparison.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  MODEL ARCHITECTURE COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        header = f"{'Model':20s}" + "".join(f"{l:>15s}" for l in metric_labels) + f"{'Time (s)':>12s}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for name in model_names:
            m = all_results[name]['metrics']
            t = all_results[name]['history']['training_time']
            row = f"{name:20s}"
            for k in metric_keys:
                row += f"{m[k]:15.4f}"
            row += f"{t:12.1f}"
            f.write(row + "\n")

    print(f"\n  Comparison saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(model_name, args, data_tuple, device, results_dir):
    """Run a single training experiment."""
    X_train, y_train, X_val, y_val, X_test, y_test, meta = data_tuple
    class_names = meta['class_names']
    num_classes = len(class_names)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {model_name}  |  scenario: {meta.get('scenario', '?')}")
    print(f"{'='*70}")

    # Create output directory
    output_dir = results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=num_classes).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Architecture: {model_class.name}")
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy()
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"  Class weights: {dict(zip(class_names, class_weights.round(4)))}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # DataLoaders (pin_memory only helps CUDA)
    pin = device.type == 'cuda'
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)

    # Train
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=args.epochs, patience=args.patience,
        use_specaugment=args.specaugment, use_amp=args.amp,
    )

    # Evaluate on validation set
    print("\n  --- Validation Set ---")
    val_metrics = evaluate_model(model, val_loader, device, class_names, "Validation")

    # Evaluate on test set
    print("\n  --- Test Set ---")
    test_metrics = evaluate_model(model, test_loader, device, class_names, "Test")

    # Generate all visualizations
    print("\n  Generating visualizations...")
    plot_training_curves(history, model_name, output_dir)
    plot_confusion_matrices(test_metrics, class_names, model_name, output_dir)
    plot_roc_curves(test_metrics, class_names, model_name, output_dir)
    plot_metrics_summary(test_metrics, class_names, model_name, output_dir)
    save_metrics_report(test_metrics, history, model_name, class_names, device, output_dir)

    # Save model
    model_path = output_dir / f'{model_name}_best.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'architecture': model_name,
        'scenario': meta.get('scenario'),
        'best_epoch': history['best_epoch'],
        'test_metrics': {k: v for k, v in test_metrics.items()
                         if k not in ('confusion_matrix', 'all_preds', 'all_labels', 'all_probs',
                                      'classification_report', 'per_class_auc')},
    }, model_path)
    print(f"  Model saved: {model_path}")

    # Scalar metrics as JSON — machine-readable for LORO aggregation
    scalar_metrics = {k: float(v) for k, v in test_metrics.items()
                      if isinstance(v, (int, float, np.floating))}
    scalar_metrics['per_class_auc'] = {k: float(v) for k, v in test_metrics['per_class_auc'].items()}
    # validation metrics too — hyperparameter tuning must select on val, not test
    scalar_metrics.update({f'val_{k}': float(v) for k, v in val_metrics.items()
                           if isinstance(v, (int, float, np.floating))})
    scalar_metrics['best_epoch'] = history['best_epoch']
    scalar_metrics['training_time'] = history['training_time']
    with open(output_dir / f'{model_name}_metrics.json', 'w') as f:
        json.dump(scalar_metrics, f, indent=2)

    # Export penultimate-layer embeddings (test set) for similarity analysis
    if args.export_embeddings:
        emb, emb_labels, emb_logits = extract_embeddings(model, test_loader, device)
        np.savez_compressed(output_dir / f'{model_name}_embeddings.npz',
                            embeddings=emb, labels=emb_labels, logits=emb_logits,
                            class_names=np.array(class_names))
        print(f"  Embeddings exported: {emb.shape}")

    return {'metrics': test_metrics, 'history': history, 'val_metrics': val_metrics}


def main():
    parser = argparse.ArgumentParser(description="Taamify Training Pipeline")
    parser.add_argument('--model', type=str, default='cnn',
                        choices=list(MODEL_REGISTRY.keys()) + ['all'],
                        help='Model architecture to train (default: cnn)')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR,
                        help='Prepared data directory (a prepared_data/<scenario> folder)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate (default: 0.0005)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='L2 regularization weight decay (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--no-specaugment', action='store_true',
                        help='Disable SpecAugment')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (AMP, CUDA only)')
    parser.add_argument('--tag', default='',
                        help='suffix for the results dir — lets hyperparameter '
                             'sweep runs coexist instead of overwriting')
    parser.add_argument('--no-embeddings', action='store_true',
                        help='Skip penultimate-layer embedding export')
    args = parser.parse_args()

    args.specaugment = not args.no_specaugment
    args.amp = not args.no_amp
    args.export_embeddings = not args.no_embeddings

    # Device: MPS (Apple Silicon) -> CUDA -> CPU
    device = pick_device()
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print("Device: Apple Silicon GPU (MPS)")
    else:
        print("WARNING: No GPU detected, using CPU")

    # Load data
    data_tuple = load_data(args.data_dir, device.type)

    # Results grouped per scenario so LORO folds don't overwrite random-split runs
    results_dir = RESULTS_ROOT / (args.data_dir.name + (f"_{args.tag}" if args.tag else ""))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    if args.model == 'all':
        models_to_train = list(MODEL_REGISTRY.keys())
    else:
        models_to_train = [args.model]

    all_results = {}
    failed = {}
    for model_name in models_to_train:
        try:
            result = run_experiment(model_name, args, data_tuple, device, results_dir)
            all_results[model_name] = result
        except Exception as e:
            # one broken architecture must not kill the whole comparison run
            import traceback
            traceback.print_exc()
            failed[model_name] = str(e)
            print(f"\n  MODEL {model_name} FAILED: {e} — continuing with next model\n",
                  flush=True)

    # Compare if multiple models
    if len(all_results) > 1:
        plot_model_comparison(all_results, results_dir)

    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    if failed:
        print(f"  FAILED models: {failed}")
    print(f"\n  Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
