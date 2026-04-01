# MmorphAttention_optimized.py

# ------------------------------------------------------------------------------
# STEP 1: Setup Kaggle and download dataset
# ------------------------------------------------------------------------------
print("[STEP 1/9] Setting up Kaggle credentials and downloading dataset ...")

import os
import subprocess
import zipfile
import shutil

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
subprocess.run(["cp", "/content/kaggle.json", os.path.expanduser("~/.kaggle/")], check=True)
subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)
print("[STEP 1/9] Kaggle credentials configured.")

KAGGLE_DATASET = "hafiznouman786/annotated-dataset-for-knee-arthritis-detection"
DOWNLOAD_DIR   = "/content"
DATA_DIR       = "./knee_images"
zip_path       = os.path.join(DOWNLOAD_DIR, "annotated-dataset-for-knee-arthritis-detection.zip")

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", KAGGLE_DATASET,
    "-p", DOWNLOAD_DIR
], check=True)
print("[STEP 1/9] Dataset downloaded.")

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(DOWNLOAD_DIR)
print("[STEP 1/9] Dataset extracted.")

if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
    extracted_root = None
    for candidate in ["Training", "training"]:
        full = os.path.join(DOWNLOAD_DIR, candidate)
        if os.path.isdir(full):
            extracted_root = full
            break
    if extracted_root:
        shutil.copytree(extracted_root, DATA_DIR)
        print(f"[STEP 1/9] Dataset copied to {DATA_DIR}")
    else:
        raise RuntimeError("Could not locate extracted dataset folder.")
else:
    print(f"[STEP 1/9] Dataset already present at {DATA_DIR}")

# ------------------------------------------------------------------------------
# STEP 2: Import Libraries
# ------------------------------------------------------------------------------
print("\n[STEP 2/9] Importing libraries ...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    subprocess.run(["pip", "install", "tqdm", "-q"], check=True)
    from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[STEP 2/9] Running on device: {DEVICE}")
print("[STEP 2/9] Libraries imported.")

# ------------------------------------------------------------------------------
# STEP 3: Hyperparameters and config
# ------------------------------------------------------------------------------
print("\n[STEP 3/9] Setting hyperparameters ...")

BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.5
PATIENCE      = 12
GRAD_CLIP     = 1.0     # gradient clipping max norm

CLASSES  = ["Osteophytes", "Sclerosis", "Erosion", "Deformity"]
SAVE_DIR = "./mmorphattention_models"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[STEP 3/9] Batch: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Patience: {PATIENCE}")
print(f"[STEP 3/9] Grad clip: {GRAD_CLIP} | Weight decay: {WEIGHT_DECAY}")
print(f"[STEP 3/9] Classes: {CLASSES}")
print(f"[STEP 3/9] Model checkpoints will be saved to: {SAVE_DIR}")

# ------------------------------------------------------------------------------
# STEP 4: Helper classes and model definition
# ------------------------------------------------------------------------------
print("\n[STEP 4/9] Defining helper classes and model architecture ...")


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.delta      = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter    = 0


class MorphologyGrader:
    """Converts model probabilities into Clinical Grading Estimates."""
    @staticmethod
    def get_grade(probability):
        if probability < 0.2: return "Grade 0 (Normal)"
        if probability < 0.4: return "Grade 1 (Doubtful)"
        if probability < 0.6: return "Grade 2 (Mild)"
        if probability < 0.8: return "Grade 3 (Moderate)"
        return "Grade 4 (Severe)"


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=32):   # ratio=32 halves internal FC dims vs original 16
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1      = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1    = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding      = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, planes, ratio=32):      # ratio passed through from ChannelAttention
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio=ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class MmorphAttention(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(MmorphAttention, self).__init__()

        # -- ResNet50: Stream 1 (Structure) ------------------------------------
        print("[STEP 4/9] Loading ResNet50 pretrained weights (Stream 1: Structure) ...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze first 6 children of ResNet50 (conv1, bn1, relu, maxpool, layer1, layer2)
        # These are low-level ImageNet features -- no need to retrain
        for param in list(self.resnet_features[:6].parameters()):
            param.requires_grad = False
        print("[STEP 4/9] ResNet50 loaded. Early layers ([:6]) frozen.")

        # -- VGG19: Stream 2 (Texture) -----------------------------------------
        print("[STEP 4/9] Loading VGG19 pretrained weights (Stream 2: Texture) ...")
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg_features = vgg.features

        # Freeze first 18 layers of VGG19 features (up to the 3rd pooling block)
        for param in list(self.vgg_features[:18].parameters()):
            param.requires_grad = False
        print("[STEP 4/9] VGG19 loaded. Early layers ([:18]) frozen.")

        feature_dim = 2048 + 512   # ResNet50 + VGG19 fusion dim

        # CBAM with ratio=32 -- faster than ratio=16
        self.attention = CBAM(feature_dim, ratio=32)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        self.gradients = None

        # Count trainable vs frozen params
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[STEP 4/9] Total params    : {total:,}")
        print(f"[STEP 4/9] Trainable params: {trainable:,}  ({100*trainable/total:.1f}% of total)")
        print(f"[STEP 4/9] Frozen params   : {total - trainable:,}  ({100*(total-trainable)/total:.1f}% of total)")

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        f_res      = self.resnet_features(x)
        f_vgg      = self.vgg_features(x)
        x_combined = torch.cat((f_res, f_vgg), dim=1)

        if x_combined.requires_grad:
            x_combined.register_hook(self.activations_hook)

        x_out  = self.attention(x_combined)
        x_out  = self.avgpool(x_out)
        logits = self.classifier(x_out)
        return logits

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        f_res = self.resnet_features(x)
        f_vgg = self.vgg_features(x)
        return torch.cat((f_res, f_vgg), dim=1)


print("[STEP 4/9] Model architecture defined.")

# ------------------------------------------------------------------------------
# STEP 5: Dataset and transforms
# ------------------------------------------------------------------------------
print("\n[STEP 5/9] Setting up dataset class and transforms ...")


class RealKneeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir    = root_dir
        self.transform   = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.png"),  recursive=True)
        self.image_paths += glob.glob(os.path.join(root_dir, "**", "*.jpg"),  recursive=True)
        self.image_paths += glob.glob(os.path.join(root_dir, "**", "*.jpeg"), recursive=True)

        if len(self.image_paths) == 0:
            print(f"[STEP 5/9] WARNING: No images found in {root_dir}")
        else:
            print(f"[STEP 5/9] Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # NOTE: Replace with real labels loaded from a CSV for production use.
        # Random labels are used here for pipeline demonstration only.
        labels = torch.randint(0, 2, (4,)).float()

        return image, labels, img_path


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[STEP 5/9] Dataset class and transforms ready.")

# ------------------------------------------------------------------------------
# STEP 6: Metric helpers and Grad-CAM
# ------------------------------------------------------------------------------
print("\n[STEP 6/9] Defining metric and Grad-CAM helper functions ...")


def calculate_metrics(outputs, labels, threshold=0.5):
    preds     = (torch.sigmoid(outputs) > threshold).float()
    tp        = (preds * labels).sum().item()
    fp        = (preds * (1 - labels)).sum().item()
    fn        = ((1 - preds) * labels).sum().item()
    tn        = ((1 - preds) * (1 - labels)).sum().item()
    epsilon   = 1e-7
    accuracy  = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall    = tp / (tp + fn + epsilon)
    f1        = 2 * (precision * recall) / (precision + recall + epsilon)
    return accuracy, f1


def generate_gradcam(model, image_tensor, target_class_index):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    output       = model(image_tensor)
    model.zero_grad()

    one_hot = torch.zeros((1, output.size()[-1]), device=DEVICE)
    one_hot[0][target_class_index] = 1
    output.backward(gradient=one_hot)

    gradients   = model.get_activations_gradient()
    activations = model.get_activations(image_tensor)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= np.max(heatmap) + 1e-7
    return heatmap


print("[STEP 6/9] Helper functions defined.")

# ------------------------------------------------------------------------------
# STEP 7: Training loop
# ------------------------------------------------------------------------------
print("\n[STEP 7/9] Defining training loop ...")


def train_model(model, loader, criterion, optimizer, scheduler, early_stopping, epochs=50):
    history = {'loss': [], 'f1': [], 'acc': []}

    best_f1        = 0.0
    best_model_pth = os.path.join(SAVE_DIR, "best_mmorphattention.pth")

    print(f"[STEP 7/9] Starting training for up to {epochs} epochs on {DEVICE} ...")
    print(f"[STEP 7/9] Best checkpoint will be saved to: {best_model_pth}")
    print("-" * 65)

    for epoch in range(epochs):
        model.train()
        run_loss, run_f1, run_acc = 0.0, 0.0, 0.0

        # -- TRAIN loader bar --------------------------------------------------
        train_bar = tqdm(
            loader,
            desc=f"  TRAIN  Ep{epoch+1:03d}/{epochs}",
            unit="batch",
            ncols=110,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

        for images, labels, _ in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping -- prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()

            run_loss += loss.item()
            acc, f1   = calculate_metrics(outputs, labels)
            run_f1   += f1
            run_acc  += acc

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.4f}",
                f1=f"{f1:.4f}"
            )

        epoch_loss = run_loss / len(loader)
        epoch_f1   = run_f1   / len(loader)
        epoch_acc  = run_acc  / len(loader)

        scheduler.step(epoch_loss)

        history['loss'].append(epoch_loss)
        history['f1'].append(epoch_f1)
        history['acc'].append(epoch_acc)

        # -- SUMMARY bar -------------------------------------------------------
        summary_bar = tqdm(
            total=0,
            desc=f"  SUMMARY Ep{epoch+1:03d}/{epochs}",
            ncols=110,
            bar_format="{desc} | {postfix}",
            leave=True
        )
        summary_bar.set_postfix(
            loss=f"{epoch_loss:.4f}",
            acc=f"{epoch_acc:.4f}",
            f1=f"{epoch_f1:.4f}",
            no_improve=f"{early_stopping.counter}/{early_stopping.patience}"
        )
        summary_bar.close()

        # Save best model by F1
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), best_model_pth)
            print(f"  [EPOCH {epoch+1:03d}] F1 improved to {best_f1:.4f}. Checkpoint saved -> {best_model_pth}")

        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print(f"\n[STEP 7/9] Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"\n[STEP 7/9] Training complete. Best F1: {best_f1:.4f}")
    return history


print("[STEP 7/9] Training loop defined.")

# ------------------------------------------------------------------------------
# STEP 8: Inference UI
# ------------------------------------------------------------------------------
print("\n[STEP 8/9] Defining inference dashboard function ...")


def run_inference_ui(model, image_path):
    """Generates a full dashboard for a single image."""
    print(f"[STEP 8/9] Running inference on: {image_path}")
    model.eval()

    img_orig   = Image.open(image_path).convert('RGB')
    transform  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_orig).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    print(f"[STEP 8/9] Prediction probabilities: "
          + "  ".join([f"{CLASSES[i]}: {probs[i]:.3f}" for i in range(len(CLASSES))]))

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.5])

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(img_orig)
    ax0.set_title("Uploaded X-Ray")
    ax0.axis('off')

    top_class_idx = np.argmax(probs)
    heatmap       = generate_gradcam(model, img_tensor, top_class_idx)
    heatmap       = cv2.resize(heatmap, (img_orig.size[0], img_orig.size[1]))
    heatmap       = np.uint8(255 * heatmap)
    heatmap       = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap       = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    img_np        = np.array(img_orig) / 255.0
    superimposed  = np.clip(heatmap * 0.4 + img_np * 0.6, 0, 1)

    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(superimposed)
    ax1.set_title(f"Morphological Signal\n(Focus: {CLASSES[top_class_idx]})")
    ax1.axis('off')

    ax2 = plt.subplot(gs[:, 2])
    ax2.axis('off')
    report_text = "=== BONE MORPHOLOGY REPORT ===\n\n"
    for i, cls in enumerate(CLASSES):
        prob  = probs[i]
        grade = MorphologyGrader.get_grade(prob)
        bar   = "X" * int(prob * 10) + "." * (10 - int(prob * 10))
        report_text += f"{cls.upper()}:\n"
        report_text += f"  Confidence: {prob*100:.1f}%\n"
        report_text += f"  Severity:   {grade}\n"
        report_text += f"  Signal:     [{bar}]\n\n"
    ax2.text(0.1, 0.9, report_text, fontsize=12, family='monospace', verticalalignment='top')

    ax3 = plt.subplot(gs[1, :2])
    bars = ax3.bar(CLASSES, probs, color='skyblue', alpha=0.7)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Probability")
    ax3.axhline(0.5, color='gray', linestyle='--')
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"inference_{os.path.basename(image_path)}.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"[STEP 8/9] Inference dashboard saved -> {out_path}")


print("[STEP 8/9] Inference dashboard function defined.")

# ------------------------------------------------------------------------------
# Main pipeline: build model, train, plot, infer
# ------------------------------------------------------------------------------
print("\n[STEP 8/9] Initializing model and dataset ...")

model   = MmorphAttention(num_classes=4, dropout_rate=DROPOUT_RATE).to(DEVICE)
dataset = RealKneeDataset(root_dir=DATA_DIR, transform=train_transform)

if len(dataset) > 0:
    print(f"[STEP 8/9] Dataset ready: {len(dataset)} images")

    # num_workers=2 and pin_memory=True for faster data loading
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True
    )
    print(f"[STEP 8/9] DataLoader ready: {len(dataloader)} batches per epoch")

    criterion      = nn.BCEWithLogitsLoss()

    # AdamW instead of Adam -- better weight decay handling
    optimizer      = optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )
    scheduler      = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = train_model(
        model, dataloader, criterion, optimizer, scheduler, early_stopping, epochs=NUM_EPOCHS
    )

    print("\n[STEP 8/9] Generating training history plots ...")
    plt.figure(figsize=(10, 4))
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['f1'],  label='F1 Score')
    plt.plot(history['loss'], label='Loss')
    plt.title("Training Performance")
    plt.xlabel("Epoch")
    plt.legend()
    plot_path = os.path.join(SAVE_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=100)
    plt.show()
    print(f"[STEP 8/9] Training plot saved -> {plot_path}")

    print("\n[STEP 8/9] Running inference demo on first image ...")
    test_img_path = dataset.image_paths[0]
    print(f"[STEP 8/9] Using image: {test_img_path}")
    run_inference_ui(model, test_img_path)

else:
    print("[STEP 8/9] ERROR: Cannot train without data. Exiting.")

# ------------------------------------------------------------------------------
# STEP 9: Save final model (.pth state dict + .pt TorchScript)
# ------------------------------------------------------------------------------
print("\n[STEP 9/9] Saving final model for meta-model use ...")

FINAL_PTH = os.path.join(SAVE_DIR, "final_mmorphattention.pth")
FINAL_PT  = os.path.join(SAVE_DIR, "final_mmorphattention.pt")

torch.save(model.state_dict(), FINAL_PTH)
print(f"[STEP 9/9] State dict saved -> {FINAL_PTH}")

try:
    model.eval()
    cpu_model = model.cpu()
    example   = torch.randn(1, 3, 224, 224)
    scripted  = torch.jit.trace(cpu_model, example)
    torch.jit.save(scripted, FINAL_PT)
    print(f"[STEP 9/9] TorchScript model saved -> {FINAL_PT}")
except Exception as e:
    print(f"[STEP 9/9] TorchScript save failed (use .pth instead): {e}")

print(f"\n[STEP 9/9] All files in {SAVE_DIR}:")
for fname in sorted(os.listdir(SAVE_DIR)):
    fpath   = os.path.join(SAVE_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname}  ({size_mb:.1f} MB)")

print("\n[DONE] Pipeline finished successfully.")