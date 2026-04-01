import os
import random
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
RF_PATH = '/content/meta_model_outputs_v2/meta_randomforest.pkl'
TEST_FOLDER = '/content/test'

# -------------------------------
# 📦 LOAD MODEL
# -------------------------------
with open(RF_PATH, 'rb') as f:
    rf_data = pickle.load(f)
    model_rf = rf_data['clf']
    scaler = rf_data['scaler']

print("✅ RandomForest model loaded")

# -------------------------------
# 📂 LOAD TEST DATA
# -------------------------------
all_test_paths = []

for root, dirs, files in os.walk(TEST_FOLDER):
    for fn in files:
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            parent_folder = os.path.basename(root)
            if parent_folder.isdigit():
                path = os.path.join(root, fn)
                all_test_paths.append((path, int(parent_folder)))

print(f"📊 Found {len(all_test_paths)} total images")

# -------------------------------
# 🎯 SAMPLE DATA
# -------------------------------
num_samples = min(500, len(all_test_paths))
selected_test = random.sample(all_test_paths, num_samples)

# -------------------------------
# 🔮 PREDICTION LOOP
# -------------------------------
y_true = []
y_pred = []
y_probs = []
errors = 0

print(f"🚀 Running inference on {num_samples} images...\n")

for img_path, label in selected_test:
    try:
        # ---- Feature Extraction ----
        feat_vec = extract_features(img_path)
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))

        # ---- Prediction ----
        pred = model_rf.predict(feat_scaled)[0]
        probs = model_rf.predict_proba(feat_scaled)[0]

        y_true.append(label)
        y_pred.append(pred)
        y_probs.append(probs)

    except Exception as e:
        errors += 1
        continue

# -------------------------------
# 📊 EVALUATION
# -------------------------------
print(f"{'='*50}")
print(f"FINAL RANDOM FOREST RESULTS")
print(f"{'='*50}")
print(f"Total Used Samples : {len(y_true)}")
print(f"Errors/Skipped     : {errors}")

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Accuracy        : {acc*100:.2f}%\n")

# Classification Report
print("📄 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=[f'Grade {i}' for i in range(5)],
    digits=3
))

# -------------------------------
# 📊 EXTRA ANALYSIS
# -------------------------------
y_probs = np.array(y_probs)

# Confidence analysis
confidences = np.max(y_probs, axis=1)

print("\n📈 Confidence Stats:")
print(f"Avg Confidence     : {np.mean(confidences)*100:.2f}%")
print(f"High Conf (>80%)   : {(confidences > 0.8).sum()} samples")
print(f"Low Conf (<50%)    : {(confidences < 0.5).sum()} samples")

# -------------------------------
# 🔍 DEBUG INSIGHT
# -------------------------------
print("\n🔍 Key Observations:")
print("- If Grade 1 recall is low → class imbalance issue")
print("- If Grade 0 is high → model bias toward healthy class")
print("- Confidence spread shows model certainty quality")