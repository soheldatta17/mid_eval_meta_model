import os, sys, subprocess
import numpy as np
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import pickle

# --- gdown ---
try:
    import gdown
except ImportError:
    subprocess.run(["pip", "install", "gdown", "-q"], check=True)
    import gdown

# ================================================================
# CONFIG
# ================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DRIVE_ASSETS = {
    "model1": {
        "id":    "1orbyJ0UU44HT3G8inoGstlJ0DhJlQXjj",
        "local": "/content/best_knee_ensemble_cbam.pt",
        "desc":  "CBAM ensemble (TorchScript)",
    },
    "model2": {
        "id":    "1Hr4gHki9nl6nmXPO0xsAU7FnlfqldHZ8",
        "local": "/content/final_knee_cnn_model.keras",
        "desc":  "Keras CNN",
    },
    "model3": {
        "id":    "16ozIZmH36J0K90bY9Jfe4YDTvS2SDDPh",
        "local": "/content/final_mmorphattention.pt",
        "desc":  "MorphAttention (TorchScript)",
    },
    "rf": {
        "id":    "1Ih1lQjV7yxYytd08a71OzEeniH4iQHxw",
        "local": "/content/meta_randomforest.pkl",
        "desc":  "RF meta-learner + scaler",
    },
}

# Paste optimized weights from the training script here.
# Leave as None to fall back to equal 1/3 weights.
OPT_W1 = None   # e.g. 0.417
OPT_W2 = None   # e.g. 0.333
OPT_W3 = None   # e.g. 0.250

SEVERITY_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
NUM_SEVERITY   = 5
META_INPUT_DIM = 14

# ================================================================
# DOWNLOAD ALL ASSETS FROM DRIVE (skips if already cached)
# ================================================================
def _gdrive_download(asset_key):
    asset = DRIVE_ASSETS[asset_key]
    local = asset["local"]
    if os.path.exists(local):
        print(f"[INFO ] {asset['desc']} already cached at {local}")
        return local
    print(f"[INFO ] Downloading {asset['desc']} from Google Drive ...")
    url = f"https://drive.google.com/uc?id={asset['id']}"
    gdown.download(url, local, quiet=False)
    if not os.path.exists(local):
        raise RuntimeError(f"[ERR  ] Download failed for {asset['desc']} — check file permissions (must be 'Anyone with link')")
    size_mb = os.path.getsize(local) / 1e6
    print(f"[OK  ] Saved to {local}  ({size_mb:.1f} MB)")
    return local

print(f"\n[START] Device: {DEVICE}")
print(f"[INFO ] Checking / downloading all model assets ...\n")

for key in ["model1", "model2", "model3", "rf"]:
    _gdrive_download(key)

# ================================================================
# LOAD MODELS
# ================================================================
print(f"\n[INFO ] Loading Model 1 (CBAM ensemble) ...")
model1 = torch.jit.load(DRIVE_ASSETS["model1"]["local"], map_location=DEVICE)
model1.eval()
print(f"[OK  ] Model 1 ready")

print(f"[INFO ] Loading Model 2 (Keras CNN) ...")
model2_tf = tf.keras.models.load_model(DRIVE_ASSETS["model2"]["local"])
model2_tf.trainable = False
try:
    _m2_last_act = model2_tf.layers[-1].activation.__name__
except Exception:
    _m2_last_act = None
MODEL2_HAS_SOFTMAX = (_m2_last_act == "softmax")
print(f"[OK  ] Model 2 ready  (final act: {_m2_last_act})")

print(f"[INFO ] Loading Model 3 (MorphAttention) ...")
model3 = torch.jit.load(DRIVE_ASSETS["model3"]["local"], map_location=DEVICE)
model3.eval()
print(f"[OK  ] Model 3 ready")

print(f"[INFO ] Loading RF meta-learner ...")
with open(DRIVE_ASSETS["rf"]["local"], "rb") as f:
    rf_data = pickle.load(f)
    rf      = rf_data["clf"]
    scaler  = rf_data["scaler"]
print(f"[OK  ] RF meta-learner + scaler ready")

# ================================================================
# TRANSFORMS
# ================================================================
transform_pt = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def _preprocess_pt(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform_pt(img).unsqueeze(0).to(DEVICE)

def _preprocess_tf(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, axis=0)

# ================================================================
# CORE PREDICTION FUNCTION
# ================================================================
def predict_single(image_path, opt_w1=OPT_W1, opt_w2=OPT_W2, opt_w3=OPT_W3):
    """
    Takes a path to a single knee X-ray image.
    Returns predicted severity grade, label, and all confidence scores.
    """

    if not os.path.exists(image_path):
        print(f"[ERR  ] File not found: {image_path}")
        return None

    print(f"\n{'='*56}")
    print(f"  IMAGE  : {os.path.basename(image_path)}")
    print(f"{'='*56}")

    # --- Per-model raw probs ---
    print(f"[INFO ] Running Model 1 ...")
    img_pt = _preprocess_pt(image_path)
    with torch.no_grad():
        probs1 = F.softmax(model1(img_pt), dim=1).cpu().numpy()[0]

    print(f"[INFO ] Running Model 2 ...")
    img_tf = _preprocess_tf(image_path)
    out2   = model2_tf.predict(img_tf, verbose=0)
    probs2 = out2[0] if MODEL2_HAS_SOFTMAX else tf.nn.softmax(out2).numpy()[0]

    print(f"[INFO ] Running Model 3 ...")
    with torch.no_grad():
        raw3   = torch.sigmoid(model3(img_pt)).cpu().numpy()[0]
    probs3 = np.zeros(NUM_SEVERITY, dtype=np.float32)
    probs3[:len(raw3)] = raw3

    # --- Per-model standalone predictions ---
    pred1 = int(np.argmax(probs1))
    pred2 = int(np.argmax(probs2))
    pred3 = int(np.argmax(probs3))

    print(f"\n  Per-model predictions:")
    print(f"  {'Model':<24} {'Grade':>6}  {'Label':<12} {'Confidence':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Model1 CBAM ensemble':<24} {'Gr'+str(pred1):>6}  {SEVERITY_NAMES[pred1]:<12} {probs1[pred1]*100:>9.2f}%")
    print(f"  {'Model2 Keras CNN':<24} {'Gr'+str(pred2):>6}  {SEVERITY_NAMES[pred2]:<12} {probs2[pred2]*100:>9.2f}%")
    print(f"  {'Model3 MorphAttention':<24} {'Gr'+str(pred3):>6}  {SEVERITY_NAMES[pred3]:<12} {probs3[pred3]*100:>9.2f}%")

    # --- Determine weights ---
    if opt_w1 is not None and opt_w2 is not None and opt_w3 is not None:
        w1, w2, w3 = opt_w1, opt_w2, opt_w3
        weight_src = "optimized"
    else:
        w1 = w2 = w3 = 1/3
        weight_src = "equal (no optimized weights set)"

    print(f"\n  Ensemble weights ({weight_src}):")
    print(f"  M1={w1:.3f}  M2={w2:.3f}  M3={w3:.3f}")

    # --- Weighted ensemble prediction ---
    ensemble_probs = w1*probs1 + w2*probs2 + w3*probs3
    ensemble_pred  = int(np.argmax(ensemble_probs))

    print(f"\n  Weighted ensemble probability distribution:")
    print(f"  {'Grade':<6} {'Label':<12} {'Prob':>8}  Bar")
    print(f"  {'-'*44}")
    for i, (name, prob) in enumerate(zip(SEVERITY_NAMES, ensemble_probs)):
        bar    = "#" * int(prob * 40)
        marker = "  <-- predicted" if i == ensemble_pred else ""
        print(f"  Gr{i:<4} {name:<12} {prob*100:>7.2f}%  {bar}{marker}")

    # --- RF meta-learner prediction ---
    print(f"\n[INFO ] Running RF meta-learner ...")
    feat_vec = np.concatenate([probs1, probs2, raw3]).astype(np.float32).reshape(1, -1)
    feat_sc  = scaler.transform(feat_vec)
    rf_pred  = int(rf.predict(feat_sc)[0])
    rf_proba = rf.predict_proba(feat_sc)[0]

    print(f"\n  RF meta-learner prediction:")
    print(f"  {'Grade':<6} {'Label':<12} {'Prob':>8}  Bar")
    print(f"  {'-'*44}")
    for i, (name, prob) in enumerate(zip(SEVERITY_NAMES, rf_proba)):
        bar    = "#" * int(prob * 40)
        marker = "  <-- predicted" if i == rf_pred else ""
        print(f"  Gr{i:<4} {name:<12} {prob*100:>7.2f}%  {bar}{marker}")

    # --- Agreement check ---
    all_preds   = [pred1, pred2, pred3, ensemble_pred, rf_pred]
    agree_count = all_preds.count(rf_pred)
    agreement   = "HIGH" if agree_count >= 4 else "MODERATE" if agree_count >= 3 else "LOW"

    print(f"\n{'='*56}")
    print(f"  FINAL PREDICTION  (RF meta-learner)")
    print(f"  Grade             : {rf_pred}")
    print(f"  Label             : {SEVERITY_NAMES[rf_pred]}")
    print(f"  RF confidence     : {rf_proba[rf_pred]*100:.2f}%")
    print(f"  Ensemble conf     : {ensemble_probs[ensemble_pred]*100:.2f}%")
    print(f"  Model agreement   : {agreement}  ({agree_count}/5 agree on Gr{rf_pred})")
    print(f"{'='*56}\n")

    return {
        "grade":          rf_pred,
        "label":          SEVERITY_NAMES[rf_pred],
        "rf_confidence":  float(rf_proba[rf_pred]),
        "rf_proba":       rf_proba.tolist(),
        "ensemble_probs": ensemble_probs.tolist(),
        "ensemble_pred":  ensemble_pred,
        "model1_pred":    pred1,
        "model2_pred":    pred2,
        "model3_pred":    pred3,
        "agreement":      agreement,
    }
    