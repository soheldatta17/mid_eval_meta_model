# KNEE THRESHOLD--V1_optimized.py


# ------------------------------------------------------------------------------
# CELL 1: Install & Setup
# ------------------------------------------------------------------------------

import os
import shutil
import subprocess
import zipfile

print("[STEP 1/12] Setting up directories and Kaggle credentials ...")

SAVE_DIR  = '/content/knee_models'
CACHE_DIR = '/content/knee_cache'    # preprocessed images cached here
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"[STEP 1/12] Model checkpoints : {SAVE_DIR}")
print(f"[STEP 1/12] Preprocessing cache: {CACHE_DIR}")

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
subprocess.run(["cp", "/content/kaggle.json", os.path.expanduser("~/.kaggle/")], check=True)
subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)
print("[STEP 1/12] Kaggle credentials configured.")

# ------------------------------------------------------------------------------
# CELL 2: Download & Extract Dataset
# ------------------------------------------------------------------------------
print("\n[STEP 2/12] Downloading and extracting dataset ...")

KAGGLE_DATASET = "hafiznouman786/annotated-dataset-for-knee-arthritis-detection"
DOWNLOAD_DIR   = "/content"
DATA_DIR       = "knee_images"
zip_path       = os.path.join(DOWNLOAD_DIR, "annotated-dataset-for-knee-arthritis-detection.zip")

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", KAGGLE_DATASET,
    "-p", DOWNLOAD_DIR
], check=True)
print("[STEP 2/12] Dataset downloaded.")

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(DOWNLOAD_DIR)
print("[STEP 2/12] Dataset extracted.")

extracted_root = None
for candidate in ["Training", "training"]:
    full = os.path.join(DOWNLOAD_DIR, candidate)
    if os.path.isdir(full):
        extracted_root = full
        break

if extracted_root and not os.path.exists(DATA_DIR):
    shutil.copytree(extracted_root, DATA_DIR)
    print(f"[STEP 2/12] Dataset copied to ./{DATA_DIR}/")
elif os.path.isdir(DATA_DIR):
    print(f"[STEP 2/12] Dataset already at ./{DATA_DIR}/")
else:
    raise RuntimeError("Could not locate extracted dataset folder.")

print(f"\n[STEP 2/12] Class folders found in '{DATA_DIR}':")
for d in sorted(os.listdir(DATA_DIR)):
    if os.path.isdir(os.path.join(DATA_DIR, d)):
        print(f"  {d}")
print("[STEP 2/12] Dataset ready.")

# ------------------------------------------------------------------------------
# CELL 3: Imports
# ------------------------------------------------------------------------------
print("\n[STEP 3/12] Loading libraries ...")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    from tqdm import tqdm
except ImportError:
    subprocess.run(["pip", "install", "tqdm", "-q"], check=True)
    from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[STEP 3/12] Running on device: {device}")
print("[STEP 3/12] All libraries loaded.")

# ------------------------------------------------------------------------------
# CELL 4: Config & Data Loading
# ------------------------------------------------------------------------------
print("\n[STEP 4/12] Loading data and configuring hyperparameters ...")

BATCH_SIZE    = 6
NUM_EPOCHS    = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-3
DROPOUT_RATE  = 0.5
PATIENCE      = 10
GRAD_CLIP     = 1.0

print(f"[STEP 4/12] Config -> Batch: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | "
      f"LR: {LEARNING_RATE} | Patience: {PATIENCE} | Grad clip: {GRAD_CLIP}")


def parse_label_from_folder(folder_name):
    import re
    m = re.match(r'^(\d+)', folder_name.strip())
    if m:
        return int(m.group(1))
    digits = re.findall(r'\d+', folder_name)
    if digits:
        return int(digits[0])
    return None


file_paths, labels = [], []

for folder in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(class_dir):
        continue
    grade = parse_label_from_folder(folder)
    if grade is None:
        print(f"[STEP 4/12] WARNING: Skipping '{folder}' -- no grade parsed.")
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_paths.append(os.path.join(class_dir, fname))
            labels.append(grade)

print(f"[STEP 4/12] Total images found: {len(file_paths)}")
count_dict = Counter(labels)
print(f"[STEP 4/12] Class distribution: {dict(sorted(count_dict.items()))}")

if len(labels) == 0:
    raise RuntimeError(f"No images found in '{DATA_DIR}'.")

NUM_CLASSES  = len(count_dict)
CLASS_IDS    = sorted(count_dict.keys())
print(f"[STEP 4/12] Detected {NUM_CLASSES} classes: {CLASS_IDS}")

total_samples        = len(labels)
class_weights        = [total_samples / (NUM_CLASSES * count_dict.get(i, 1)) for i in CLASS_IDS]
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"[STEP 4/12] Class weights: {[round(w, 3) for w in class_weights]}")

label_map = {orig: idx for idx, orig in enumerate(CLASS_IDS)}
labels    = [label_map[l] for l in labels]

train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.20, stratify=labels, random_state=42
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.20,
    stratify=train_val_labels, random_state=42
)
print(f"[STEP 4/12] Split -> Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")
print("[STEP 4/12] Data loading complete.")

# ------------------------------------------------------------------------------
# CELL 5: ONE-TIME Preprocessing Cache
# Run CLAHE + edge extraction ONCE for every image and save as .png to disk.
# During training, Dataset just loads the cached file -- no CV2 work per epoch.
# ------------------------------------------------------------------------------
print("\n[STEP 5/12] Building preprocessing cache (runs once, saves to disk) ...")


def preprocess_image_cv2(image_path):
    """Returns (Edge Map BGR, CLAHE-enhanced Grayscale BGR)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load: {image_path}")
    img      = cv2.resize(img, (224, 224))
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur     = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges    = cv2.Canny(blur, 20, 80)
    return (cv2.cvtColor(edges,    cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))


def get_cache_path(original_path):
    """Returns the cache file path for a given original image path."""
    # Flatten the path to a safe filename
    safe_name = original_path.replace("/", "_").replace("\\", "_").lstrip("_")
    return os.path.join(CACHE_DIR, safe_name)


def build_cache(all_paths):
    """Preprocess all images once and save enhanced versions to CACHE_DIR."""
    already_cached = 0
    newly_cached   = 0

    cache_bar = tqdm(
        all_paths,
        desc="  CACHE  Building",
        unit="img",
        ncols=110,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    for path in cache_bar:
        cache_path = get_cache_path(path)
        if os.path.exists(cache_path):
            already_cached += 1
            cache_bar.set_postfix(cached=already_cached, new=newly_cached)
            continue
        try:
            _, enhanced = preprocess_image_cv2(path)
            cv2.imwrite(cache_path, enhanced)
            newly_cached += 1
        except Exception as e:
            print(f"\n  [CACHE] WARNING: Failed to cache {path}: {e}")
        cache_bar.set_postfix(cached=already_cached, new=newly_cached)

    print(f"[STEP 5/12] Cache complete. "
          f"Newly cached: {newly_cached}  Already existed: {already_cached}  "
          f"Total: {already_cached + newly_cached}")


# Build cache for all splits at once
all_paths = train_paths + val_paths + test_paths
build_cache(all_paths)
print("[STEP 5/12] Preprocessing cache ready.")

# ------------------------------------------------------------------------------
# CELL 6: Dataset & DataLoaders (loads from cache -- no CV2 per epoch)
# ------------------------------------------------------------------------------
print("\n[STEP 6/12] Setting up datasets and DataLoaders ...")


class KneeOADataset(Dataset):
    """
    Loads preprocessed (CLAHE-enhanced) images directly from cache.
    No CV2 processing at runtime -- just imread + transform.
    """
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        cache_path = get_cache_path(self.paths[idx])
        enhanced   = cv2.imread(cache_path)
        if enhanced is None:
            # Fallback: reprocess on the fly if cache somehow missing
            _, enhanced = preprocess_image_cv2(self.paths[idx])
        img_tensor = (self.transform(enhanced) if self.transform
                      else transforms.functional.to_tensor(enhanced))
        return img_tensor, self.labels[idx]


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = KneeOADataset(train_paths, train_labels, transform=train_transforms)
val_ds   = KneeOADataset(val_paths,   val_labels,   transform=val_transforms)
test_ds  = KneeOADataset(test_paths,  test_labels,  transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"[STEP 6/12] Datasets -> Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
print(f"[STEP 6/12] Batches  -> Train: {len(train_loader)}  Val: {len(val_loader)}")
print("[STEP 6/12] DataLoaders ready.")

# ------------------------------------------------------------------------------
# CELL 7: Model Definition (CBAM + EfficientNet-B5 + ResNet18)
# ------------------------------------------------------------------------------
print("\n[STEP 7/12] Building model (CBAM + EfficientNet-B5 + ResNet18 Ensemble) ...")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1   = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class EnsembleKnee(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        print("[STEP 7/12] Loading EfficientNet-B5 pretrained weights ...")
        try:
            self.effnet = models.efficientnet_b5(weights='DEFAULT')
        except Exception:
            self.effnet = models.efficientnet_b5(pretrained=True)
        self.eff_features = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Identity()
        self.eff_attention = CBAM(self.eff_features)

        # Freeze early EfficientNet blocks (low-level ImageNet features)
        for param in list(self.effnet.features[:4].parameters()):
            param.requires_grad = False
        print("[STEP 7/12] EfficientNet-B5 loaded. Early blocks [:4] frozen.")

        print("[STEP 7/12] Loading ResNet18 pretrained weights ...")
        try:
            self.resnet = models.resnet18(weights='DEFAULT')
        except Exception:
            self.resnet = models.resnet18(pretrained=True)
        self.res_features = self.resnet.fc.in_features
        self.resnet.fc    = nn.Identity()
        self.res_attention = CBAM(self.res_features)

        # Freeze early ResNet18 layers
        for param in list(self.resnet.parameters())[:20]:
            param.requires_grad = False
        print("[STEP 7/12] ResNet18 loaded. Early layers frozen.")

        total_feat = self.eff_features + self.res_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(total_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(512, num_classes)
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p   = sum(p.numel() for p in self.parameters())
        print(f"[STEP 7/12] Fusion: {total_feat} -> 512 -> {num_classes}")
        print(f"[STEP 7/12] Trainable params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

    def forward(self, x):
        x1 = self.effnet.features(x)
        x1 = self.effnet.avgpool(x1)
        x1 = self.eff_attention(x1)
        x1 = torch.flatten(x1, 1)

        x2 = self.resnet.conv1(x)
        x2 = self.resnet.bn1(x2)
        x2 = self.resnet.relu(x2)
        x2 = self.resnet.maxpool(x2)
        x2 = self.resnet.layer1(x2)
        x2 = self.resnet.layer2(x2)
        x2 = self.resnet.layer3(x2)
        x2 = self.resnet.layer4(x2)
        x2 = self.resnet.avgpool(x2)
        x2 = self.res_attention(x2)
        x2 = torch.flatten(x2, 1)

        return self.classifier(torch.cat((x1, x2), dim=1))


model     = EnsembleKnee(num_classes=NUM_CLASSES).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                  factor=0.5, patience=3)

print(f"[STEP 7/12] Model on {device} | Optimizer: AdamW | LR: {LEARNING_RATE}")
print("[STEP 7/12] Model ready.")

# ------------------------------------------------------------------------------
# CELL 8: Training Loop
# ------------------------------------------------------------------------------
print(f"\n[STEP 8/12] Starting training for {NUM_EPOCHS} epochs ...")
print(f"[STEP 8/12] Checkpoints -> {SAVE_DIR}")
print(f"[STEP 8/12] Early stopping patience: {PATIENCE}")
print("-" * 70)

LOCAL_PTH = os.path.join(SAVE_DIR, 'best_knee_ensemble_cbam.pth')
LOCAL_PT  = os.path.join(SAVE_DIR, 'best_knee_ensemble_cbam.pt')

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []
best_val_loss     = float('inf')
epochs_no_improve = 0
best_model_wts    = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    print(f"\n[EPOCH {epoch+1:03d}/{NUM_EPOCHS}]")

    # -- TRAIN bar -------------------------------------------------------------
    model.train()
    running_loss = correct = total = 0

    train_bar = tqdm(
        train_loader,
        desc=f"  TRAIN  Ep{epoch+1:03d}/{NUM_EPOCHS}",
        unit="batch", ncols=110, leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    for inputs, lbls in train_bar:
        inputs, lbls = inputs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
        train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    epoch_loss = running_loss / len(train_ds)
    epoch_acc  = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # -- VAL bar ---------------------------------------------------------------
    model.eval()
    running_loss = correct = total = 0

    val_bar = tqdm(
        val_loader,
        desc=f"  VAL    Ep{epoch+1:03d}/{NUM_EPOCHS}",
        unit="batch", ncols=110, leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    with torch.no_grad():
        for inputs, lbls in val_bar:
            inputs, lbls = inputs.to(device), lbls.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, lbls)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
            val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    val_loss = running_loss / len(val_ds)
    val_acc  = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    scheduler.step(val_loss)

    # -- SUMMARY bar -----------------------------------------------------------
    summary_bar = tqdm(
        total=0, desc=f"  SUMMARY Ep{epoch+1:03d}/{NUM_EPOCHS}",
        ncols=110, bar_format="{desc} | {postfix}", leave=True
    )
    summary_bar.set_postfix(
        train_loss=f"{epoch_loss:.4f}", train_acc=f"{epoch_acc:.4f}",
        val_loss=f"{val_loss:.4f}",   val_acc=f"{val_acc:.4f}",
        no_improve=f"{epochs_no_improve}/{PATIENCE}"
    )
    summary_bar.close()

    # -- Save best -------------------------------------------------------------
    if val_loss < best_val_loss:
        print(f"  [EPOCH {epoch+1:03d}] Val loss improved: {best_val_loss:.4f} -> {val_loss:.4f}. Saving ...")
        best_val_loss  = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        torch.save(model.state_dict(), LOCAL_PTH)
        try:
            cpu_model = copy.deepcopy(model).cpu().eval()
            example   = torch.randn(1, 3, 224, 224)
            scripted  = torch.jit.trace(cpu_model, example)
            torch.jit.save(scripted, LOCAL_PT)
            print(f"  [EPOCH {epoch+1:03d}] Saved -> {LOCAL_PTH}  and  {LOCAL_PT}")
        except Exception as e:
            print(f"  [EPOCH {epoch+1:03d}] State-dict saved -> {LOCAL_PTH}  (TorchScript skipped: {e})")
    else:
        epochs_no_improve += 1
        print(f"  [EPOCH {epoch+1:03d}] No improvement. Patience: {epochs_no_improve}/{PATIENCE}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n[STEP 8/12] Early stopping at epoch {epoch+1}.")
        break

print(f"\n[STEP 8/12] Training complete. Best Val Loss: {best_val_loss:.4f}")
model.load_state_dict(best_model_wts)
print("[STEP 8/12] Best weights loaded.")

FINAL_PTH = os.path.join(SAVE_DIR, 'final_knee_ensemble_cbam.pth')
torch.save(model.state_dict(), FINAL_PTH)
print(f"[STEP 8/12] Final weights saved -> {FINAL_PTH}")

print(f"\n[STEP 8/12] Files in {SAVE_DIR}:")
for f in os.listdir(SAVE_DIR):
    size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")

# ------------------------------------------------------------------------------
# CELL 9: Training Plots
# ------------------------------------------------------------------------------
print("\n[STEP 9/12] Generating training curve plots ...")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='steelblue')
plt.plot(val_losses,   label='Val Loss',   color='orange')
plt.title('Loss Curves'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc', color='green')
plt.plot(val_accs,   label='Val Acc',   color='red')
plt.title('Accuracy Curves'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
plt.show()
print(f"[STEP 9/12] Saved -> {os.path.join(SAVE_DIR, 'training_curves.png')}")

# ------------------------------------------------------------------------------
# CELL 10: Evaluation (Confusion Matrix + ROC)
# ------------------------------------------------------------------------------
print("\n[STEP 10/12] Running evaluation on test set ...")

model.eval()
all_preds, all_labels_list, all_probs = [], [], []

eval_bar = tqdm(
    test_loader, desc="  EVAL  ", unit="batch", ncols=110, leave=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)

with torch.no_grad():
    for inputs, lbls in eval_bar:
        inputs  = inputs.to(device)
        outputs = model(inputs)
        probs   = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(lbls.numpy())
        all_probs.extend(probs.cpu().numpy())

print(f"[STEP 10/12] Evaluation done. Total test samples: {len(all_preds)}")

print("[STEP 10/12] Generating confusion matrix ...")
cm   = confusion_matrix(all_labels_list, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_IDS)
plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title('Confusion Matrix')
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()
print(f"[STEP 10/12] Saved -> {os.path.join(SAVE_DIR, 'confusion_matrix.png')}")

print("[STEP 10/12] Generating ROC curves ...")
y_test_bin = label_binarize(all_labels_list, classes=list(range(NUM_CLASSES)))
all_probs  = np.array(all_probs)
colors     = cycle(['blue', 'red', 'green', 'orange', 'purple'])

plt.figure(figsize=(10, 8))
for i, color in zip(range(NUM_CLASSES), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'Grade {CLASS_IDS[i]} (AUC = {roc_auc_val:.2f})')
    print(f"  [STEP 10/12] Grade {CLASS_IDS[i]} AUC: {roc_auc_val:.4f}")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve'); plt.legend(loc='lower right')
plt.savefig(os.path.join(SAVE_DIR, 'roc_curves.png'), dpi=150)
plt.show()
print(f"[STEP 10/12] Saved -> {os.path.join(SAVE_DIR, 'roc_curves.png')}")

print("\n[STEP 10/12] Classification Report:")
print(classification_report(all_labels_list, all_preds,
                            target_names=[f'Grade {CLASS_IDS[i]}' for i in range(NUM_CLASSES)]))

# ------------------------------------------------------------------------------
# CELL 11: Heuristic Gap Measure + Fusion Prediction
# ------------------------------------------------------------------------------
print("\n[STEP 11/12] Defining heuristic gap measure and fusion prediction ...")


def measure_gap(edges_gray):
    h, w    = edges_gray.shape
    y_start = int(h * 0.35); y_end = int(h * 0.65)
    x_start = int(w * 0.40); x_end = int(w * 0.60)
    gaps, gap_coords = [], []
    for x in range(x_start, x_end, 2):
        col          = edges_gray[y_start:y_end, x]
        edge_indices = np.where(col > 0)[0] + y_start
        if len(edge_indices) < 2:
            continue
        diffs    = np.diff(edge_indices)
        max_idx  = np.argmax(diffs)
        gap_size = diffs[max_idx]
        if 2 < gap_size < (h * 0.3):
            gaps.append(gap_size)
            gap_coords.append((edge_indices[max_idx], edge_indices[max_idx + 1], x))
    if not gaps:
        return 0, (0, 0, w // 2)
    median_gap = np.median(gaps)
    best_idx   = np.argmin(np.abs(np.array(gaps) - median_gap))
    return median_gap, gap_coords[best_idx]


def annotate_and_predict(image_path):
    """Run dual-path (DL + heuristic) inference on one image."""
    img_bgr_edges, img_bgr_enhanced = preprocess_image_cv2(image_path)
    edges_gray = cv2.cvtColor(img_bgr_edges, cv2.COLOR_BGR2GRAY)
    gap, (y1, y2, xcol) = measure_gap(edges_gray)

    max_expected_gap = 18.0
    gap_score = 0.01 + (min(gap, max_expected_gap) / max_expected_gap) * 99.98

    if   gap_score < 15: h_grade = NUM_CLASSES - 1
    elif gap_score < 30: h_grade = min(3, NUM_CLASSES - 1)
    elif gap_score < 50: h_grade = min(2, NUM_CLASSES - 1)
    elif gap_score < 70: h_grade = min(1, NUM_CLASSES - 1)
    else:                h_grade = 0

    img_tensor = val_transforms(img_bgr_enhanced).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits   = model(img_tensor)
        dl_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    heuristic_probs          = np.zeros(NUM_CLASSES)
    heuristic_probs[h_grade] = 1.0
    fused_probs  = 0.7 * dl_probs + 0.3 * heuristic_probs
    final_idx    = np.argmax(fused_probs)
    confidence   = fused_probs[final_idx]
    final_grade  = CLASS_IDS[final_idx]

    ann = img_bgr_edges.copy()
    if gap > 0:
        cv2.line(ann, (xcol, y1), (xcol, y2),     (255, 0, 0), 2)
        cv2.line(ann, (xcol-8, y1), (xcol+8, y1), (255, 0, 0), 2)
        cv2.line(ann, (xcol-8, y2), (xcol+8, y2), (255, 0, 0), 2)

    return (cv2.cvtColor(img_bgr_enhanced, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(ann,              cv2.COLOR_BGR2RGB),
            {'final_grade': final_grade,
             'dl_grade':    CLASS_IDS[int(np.argmax(dl_probs))],
             'heu_grade':   CLASS_IDS[h_grade],
             'gap_score':   gap_score,
             'confidence':  float(confidence),
             'pixel_gap':   gap})


print("[STEP 11/12] Heuristic and fusion functions defined.")

# ------------------------------------------------------------------------------
# CELL 12: Automatic Inference on Random Test Samples
# ------------------------------------------------------------------------------
NUM_SAMPLES = 10

print(f"\n[STEP 12/12] Running inference on {NUM_SAMPLES} random test images ...")
print("-" * 55)

sample_paths  = random.sample(test_paths, min(NUM_SAMPLES, len(test_paths)))
sample_labels = [test_labels[test_paths.index(p)] for p in sample_paths]

correct_count = 0
for sample_idx, (img_path, true_idx) in enumerate(zip(sample_paths, sample_labels), 1):
    true_grade = CLASS_IDS[true_idx]
    print(f"\n[STEP 12/12] Sample {sample_idx}/{NUM_SAMPLES}: "
          f"{os.path.basename(img_path)} | True Grade: {true_grade}")
    try:
        dl_view, heu_view, res = annotate_and_predict(img_path)
        is_correct  = res['final_grade'] == true_grade
        result_str  = "CORRECT" if is_correct else "WRONG"
        if is_correct:
            correct_count += 1

        print(f"[STEP 12/12] Sample {sample_idx}/{NUM_SAMPLES}: {result_str} | "
              f"True: Grade {true_grade}  Predicted: Grade {res['final_grade']} "
              f"(Conf: {res['confidence']:.1%}) | "
              f"DL: G{res['dl_grade']}  Heuristic: G{res['heu_grade']} "
              f"(gap {res['pixel_gap']:.1f}px)")

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(dl_view, cmap='gray')
        axs[0].set_title("Path A: DL Input (Enhanced)", color='blue', fontweight='bold')
        axs[0].axis('off')
        axs[1].imshow(heu_view)
        axs[1].set_title("Path B: Heuristic (Edge Gap)", color='green', fontweight='bold')
        axs[1].axis('off')

        result_text = (
            f"[{result_str}]  TRUE: Grade {true_grade}  |  "
            f"PREDICTED: Grade {res['final_grade']} ({res['confidence']:.1%})\n"
            f"{'---' * 16}\n"
            f"Deep Learning: G{res['dl_grade']}   "
            f"Geometric: G{res['heu_grade']} (gap {res['pixel_gap']:.1f}px)"
        )
        plt.suptitle(result_text, fontsize=12, fontweight='bold', y=1.04)
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"inference_{os.path.basename(img_path)}")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"[STEP 12/12] Sample {sample_idx} plot saved -> {save_path}")

    except Exception as e:
        print(f"[STEP 12/12] ERROR processing {img_path}: {e}")

print(f"\n{'=' * 55}")
print(f"[STEP 12/12] Inference complete: {correct_count}/{NUM_SAMPLES} correct")
print(f"[STEP 12/12] Sample accuracy: {correct_count/NUM_SAMPLES*100:.1f}%")
print(f"[STEP 12/12] All plots saved to: {SAVE_DIR}")
print(f"{'=' * 55}")

print(f"\n[DONE] All files in {SAVE_DIR}:")
for f in sorted(os.listdir(SAVE_DIR)):
    size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")
print("[DONE] Pipeline finished successfully.")