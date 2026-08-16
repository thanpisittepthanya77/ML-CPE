import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 0. สร้างโฟลเดอร์ images หากยังไม่มี
os.makedirs('images', exist_ok=True)

# 1. โหลดข้อมูล Kaggle Iris Dataset (iris.csv หรือ Iris.csv)
csv_file = 'iris.csv' if os.path.exists('iris.csv') else 'Iris.csv'

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    # ดาวน์โหลดผ่าน URL สำรองกรณีไม่มีไฟล์ในเครื่อง
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    df.insert(0, 'Id', range(1, len(df) + 1))
    df.to_csv('iris.csv', index=False)

print("--- ข้อมูลเบื้องต้น (Dataset Overview) ---")
print(df.head())

# ลบคอลัมน์ Id ออกเนื่องจากไม่ใช่ Feature ในการเทรน
if 'Id' in df.columns or 'id' in df.columns:
    df_features = df.drop(columns=[col for col in df.columns if col.lower() == 'id'])
else:
    df_features = df.copy()

# แยก Features (X) และ Target (y)
X = df_features.iloc[:, :-1].values
y = df_features.iloc[:, -1].values

# 2. แบ่งชุดข้อมูล Train / Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize Features ก่อนการเทรน
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. เทรนและประเมินผล KNN กับค่า k ต่างๆ (k = 3, 5, 7)
k_values = [3, 5, 7]
accuracy_scores = {}

print("\n--- ผลการทดสอบโมเดล KNN ---")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[k] = acc
    print(f"Accuracy for k = {k}: {acc * 100:.2f}%")

# ค้นหาค่า k ที่ดีที่สุด
best_k = max(accuracy_scores, key=accuracy_scores.get)
print(f"\nBest k value: {best_k} (Accuracy: {accuracy_scores[best_k] * 100:.2f}%)")

# 5. ประเมินผลโมเดลที่ดีที่สุดและสร้างกราฟบันทึกลงใน images/
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_best = best_knn.predict(X_test_scaled)

print("\n--- Classification Report (Best k) ---")
print(classification_report(y_test, y_pred_best))

# บันทึกกราฟเปรียบเทียบค่า Accuracy ของแต่ละ k
plt.figure(figsize=(6, 4))
plt.plot(list(accuracy_scores.keys()), [v * 100 for v in accuracy_scores.values()], 
         marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
plt.title('KNN Classification Accuracy for different k values', fontsize=12)
plt.xlabel('k value (Number of Neighbors)', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.xticks(k_values)
plt.ylim([80, 105])
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/k_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# บันทึก Confusion Matrix
unique_labels = np.unique(y)
cm = confusion_matrix(y_test, y_pred_best, labels=unique_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title(f'Confusion Matrix (k = {best_k})', fontsize=12)
plt.xlabel('Predicted Species', fontsize=10)
plt.ylabel('Actual Species', fontsize=10)
plt.savefig('images/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("บันทึกกราฟผลลัพธ์ลงในโฟลเดอร์ images/ เรียบร้อยแล้ว")