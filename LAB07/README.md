# 🌿 LAB 07: 1D CNN Classification on Iris Dataset

การจัดทำมอดูลและประเมินประสิทธิภาพแบบจำลอง **1D Convolutional Neural Network (1D CNN)** สำหรับจำแนกสายพันธุ์ดอกไอริส (Iris Dataset) ตามข้อกำหนดใบปฏิบัติการ ML-LAB-07

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB07/
├── Iris.csv              # ชุดข้อมูล Iris Dataset (Kaggle)
├── data_prep.py          # มอดูลจัดการข้อมูล และแปลงมิติข้อมูล
├── model.py              # มอดูลนิยามโครงสร้างแบบจำลอง 1D CNN
├── evaluate.py           # มอดูลวัดผล แสดงกราฟการเรียนรู้ และทำนายผล
├── main.py               # ไฟล์หลักสำหรับสั่งรันโปรแกรมทั้งหมด
├── config_a_results.png  # กราฟแสดงผลการเรียนรู้ของ Config A
├── config_b_results.png  # กราฟแสดงผลการเรียนรู้ของ Config B
└── README.md             # เอกสารอธิบายรายละเอียดโปรเจกต์

การจัดการข้อมูล (Data Preprocessing)
คุณลักษณะของชุดข้อมูลประกอบด้วย 4 ฟีเจอร์:

SepalLengthCm — ความยาวกลีบเลี้ยง

SepalWidthCm — ความกว้างกลีบเลี้ยง

PetalLengthCm — ความยาวกลีบดอก

PetalWidthCm — ความกว้างกลีบดอก

ขั้นตอนใน data_prep.py
Data Cleaning: ตัดคอลัมน์ Id ออกจากชุดข้อมูล

Label Encoding: แปลงคลาสเป้าหมาย (Species) เป็นรหัสตัวเลข [0, 1, 2]

Train/Test Split: แบ่งข้อมูลสำหรับ Train 80% และ Test 20%

Standardization: สเกลข้อมูลฟีเจอร์ด้วย StandardScaler

Reshaping: ปรับมิติข้อมูลเป็น (Samples, 4, 1) เพื่อรองรับเลเยอร์ Conv1D


โครงสร้างแบบจำลอง (Model Architecture)
แบบจำลอง 1D CNN ออกแบบโดยไม่ใช้ Max Pooling เพื่อป้องกันการสูญเสียข้อมูลสำคัญในชุดข้อมูลขนาดเล็ก:
Layer (Type),Output Shape,Param #,Details
Conv1D,"(None, 3, filters)",48 / 96,"Kernel Size = 2, Activation = relu"
Flatten,"(None, 3 * filters)",0,แปลงมิติเป็น 1D Vector
Dense,"(None, 32)",Dynamic,Activation = relu
Dense (Output),"(None, 3)",99,Activation = softmax
Optimizer: Adam (Learning Rate = 0.005)

Loss Function: Sparse Categorical Crossentropy

Metrics: Accuracy


การทดลองเปรียบเทียบ (Configurations)
เปรียบเทียบประสิทธิภาพแบบจำลองใน 2 คอนฟิกูเรชัน:

Config A: 16 Filters | 50 Epochs | Batch Size 8

Config B: 32 Filters | 100 Epochs | Batch Size 8


ผลการทดลองและการวัดผล (Experimental Results)
1. ตารางเปรียบเทียบประสิทธิภาพ (Performance Metrics)
Configuration,Filters,Epochs,Test Accuracy,Test Loss
Config A,16,50,96.67%,0.1172
Config B,32,100,100.00%,0.0215
2. กราฟแสดงการเรียนรู้ (Training & Validation Graphs)
🔹 Config A (16 Filters, 50 Epochs)
🔹 Config B (32 Filters, 100 Epochs)
3. ตัวอย่างการทำนายผล (Sample Predictions)
================ Config A (16 Filters, 50 Epochs) ================
Test Accuracy: 96.67% | Test Loss: 0.1172

--- Sample Predictions ---
Sample 1: Actual = Iris-setosa     | Predicted = Iris-setosa
Sample 2: Actual = Iris-versicolor | Predicted = Iris-versicolor
Sample 3: Actual = Iris-virginica  | Predicted = Iris-virginica
Sample 4: Actual = Iris-versicolor | Predicted = Iris-versicolor
Sample 5: Actual = Iris-setosa     | Predicted = Iris-setosa
==================================================================

วิธีการรันโปรแกรม (How to Run)
1.ติดตั้งแพ็กเกจที่จำเป็น:
pip install tensorflow pandas numpy scikit-learn matplotlib
2.สั่งรันไฟล์หลัก:
python ML-CPE/LAB07/main.py
## 📈 ผลการทดลองและการวัดผล (Experimental Results)

### 1. ตารางเปรียบเทียบประสิทธิภาพ (Performance Metrics)

| Configuration | Filters | Epochs | Test Accuracy | Test Loss | Remark |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A** | 16 | 50 | 96.67% | 0.1172 | Optimal Fit (เสถียรที่สุด) |
| **Config B** | 32 | 100 | 100.00% | 0.0215 | Overfitting ในช่วงท้าย |

---

### 2. กราฟแสดงการเรียนรู้ (Training & Validation Graphs)

#### 🔹 Config A (16 Filters, 50 Epochs)
![Config A Results](config_a_results.png)

#### 🔹 Config B (32 Filters, 100 Epochs)
![Config B Results](config_b_results.png)