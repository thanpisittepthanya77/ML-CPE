import os
import re
import glob
import zipfile

# บังคับใช้ TkAgg Backend เพื่อให้หน้าต่างกราฟเด้งแสดงผลบน Windows/VS Code
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================================
# 0. แตกไฟล์ archive.zip อัตโนมัติ (หากยังไม่ได้แตกไฟล์)
# =========================================================================
if os.path.exists('archive.zip') and not os.path.exists('./FGNET'):
    print("กำลังแตกไฟล์ archive.zip ...")
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("แตกไฟล์สำเร็จ!\n")

# =========================================================================
# 1. โหลดข้อมูลรูปภาพ FG-NET (จำกัด 100 รูป)
# =========================================================================
dataset_path = "./FGNET/images"
MAX_SAMPLES = 1000

# หากโครงสร้างไม่มีโฟลเดอร์ images ย่อย ให้ค้นหาในโฟลเดอร์หลัก
if not os.path.exists(dataset_path):
    dataset_path = "./FGNET"

image_paths = glob.glob(os.path.join(dataset_path, "**", "*.JPG"), recursive=True) + \
              glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)

np.random.seed(42)
if len(image_paths) > MAX_SAMPLES:
    image_paths = list(np.random.choice(image_paths, MAX_SAMPLES, replace=False))

X_pixels = []
y_ages = []
img_size = (64, 64)

for p in image_paths:
    filename = os.path.basename(p)
    match = re.search(r'[Aa](\d+)', filename)
    if match:
        age = int(match.group(1))
        try:
            img = Image.open(p).convert('L').resize(img_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            X_pixels.append(img_array.flatten())
            y_ages.append(age)
        except Exception:
            continue

X_pixels = np.array(X_pixels)
y_ages = np.array(y_ages)

# แบ่ง Train 80% / Test 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_pixels, y_ages, test_size=0.2, random_state=42
)

print(f"--- โหลดข้อมูลสำเร็จ: ทั้งหมด {len(X_pixels)} ภาพ (Train: {len(X_train)}, Test: {len(X_test)}) ---\n")

# =========================================================================
# ข้อ 1: Simple Linear Regression (1 Feature)
# =========================================================================
X_train_simple = X_train.mean(axis=1).reshape(-1, 1)
X_test_simple = X_test.mean(axis=1).reshape(-1, 1)

simple_model = LinearRegression().fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)

print("=== 1. Simple Linear Regression ===")
print(f"Slope (b1): {simple_model.coef_[0]:.4f}")
print(f"Intercept (b0): {simple_model.intercept_:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_simple):.2f} ปี")
print(f"MSE: {mean_squared_error(y_test, y_pred_simple):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_simple):.4f}\n")

# =========================================================================
# ข้อ 2: Multiple Linear Regression (10 Features)
# =========================================================================
np.random.seed(42)
selected_indices = np.random.choice(X_train.shape[1], 10, replace=False)

X_train_multi = X_train[:, selected_indices]
X_test_multi = X_test[:, selected_indices]

multi_model = LinearRegression().fit(X_train_multi, y_train)
y_pred_multi = multi_model.predict(X_test_multi)

print("=== 2. Multiple Linear Regression ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred_multi):.2f} ปี")
print(f"MSE: {mean_squared_error(y_test, y_pred_multi):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_multi):.4f}\n")

# =========================================================================
# ข้อ 3: Age Prediction (PCA + Ridge Regression)
# =========================================================================
n_components = min(15, len(X_train) - 1)
pca = PCA(n_components=n_components, random_state=42)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

age_model = Ridge(alpha=1.0).fit(X_train_pca, y_train)
y_pred_age = age_model.predict(X_test_pca)

print("=== 3. Age Prediction (PCA + Ridge) ===")
print(f"Explained Variance Ratio (PCA): {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"MAE: {mean_absolute_error(y_test, y_pred_age):.2f} ปี")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_age)):.2f} ปี")
print(f"R2 Score: {r2_score(y_test, y_pred_age):.4f}\n")

# =========================================================================
# Plot กราฟแสดงผลลัพธ์ข้อ 3
# =========================================================================
print("กำลังเปิดหน้าต่างแสดงกราฟ...")
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_age, color='green', alpha=0.7, label='Predicted Age')
plt.plot([y_ages.min(), y_ages.max()], [y_ages.min(), y_ages.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Age (Years)')
plt.ylabel('Predicted Age (Years)')
plt.title('Actual vs Predicted Age (PCA + Ridge Regression)')
plt.legend()
plt.grid(True)

# ค้างหน้าต่างกราฟไว้จนกว่าผู้ใช้จะกดปิด
plt.show(block=True)