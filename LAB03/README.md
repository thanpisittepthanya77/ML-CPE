# Lab 3: Regression & Classification (FG-NET Dataset)

งานปฏิบัติการที่ 3 วิชา Machine Learning (04-624-201) มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)

## 📌 วัตถุประสงค์ (Objectives)
1. ดึงคุณลักษณะและลดมิติของข้อมูลภาพใบหน้า (Feature Extraction & Reduction) ด้วย **PCA (Principal Component Analysis)**
2. สร้างแบบจำลอง **Ridge Regression** เพื่อทำนายอายุ (Age Prediction) จากภาพใบหน้า
3. ประเมินประสิทธิภาพแบบจำลองด้วยค่าสถิติต่างๆ และแสดงผลเปรียบเทียบระหว่างค่าจริงและค่าทำนาย

## 📊 ตัววัดผลการทดลอง (Evaluation Metrics)
* **MAE (Mean Absolute Error):** วัดความคลาดเคลื่อนเฉลี่ย
* **RMSE (Root Mean Squared Error):** วัดความคลาดเคลื่อนยกกำลังสองเฉลี่ย
* **R² Score:** วัดระดับการอธิบายความแปรปรวนของโมเดล

## 📈 ผลการประเมินและการวิเคราะห์ (Results & Analysis)
* จากการพิจารณากราฟกระจายตัว (Scatter Plot) ของ **Actual vs Predicted Age** พบสภาวะ **Mean Regression Bias**:
  * โมเดลทำนายอายุสูงกว่าความเป็นจริงในกลุ่มเด็ก (Overestimation)
  * โมเดลทำนายอายุน้อยกว่าความเป็นจริงในกลุ่มผู้สูงอายุ (Underestimation)

## 📁 Dataset Reference & Credit
* **Dataset:** FG-NET Facial Age Dataset
* **Source:** Kaggle ([https://www.kaggle.com/datasets/aiolapo/fgnet-dataset](https://www.kaggle.com/datasets/aiolapo/fgnet-dataset))
* **Description:** ชุดข้อมูลภาพใบหน้า FG-NET ที่ใช้สำหรับการฝึกสอนและประเมินประสิทธิภาพของแบบจำลองในการทำนายอายุและจำแนกเพศ