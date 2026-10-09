import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_prep_data(csv_path="Iris.csv"):
    # --- ส่วนที่ต้องเพิ่ม ---
    # หาตำแหน่งโฟลเดอร์ที่ไฟล์ data_prep.py บันทึกอยู่ (ก็คือ LAB07)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # นำตำแหน่งโฟลเดอร์มาต่อกับชื่อไฟล์ Iris.csv
    real_csv_path = os.path.join(base_dir, csv_path)
    
    # --- แก้ให้โหลดจาก real_csv_path ---
    df = pd.read_csv(real_csv_path)
    
    # ลบคอลัมน์ Id ถ้ามี
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    X = df.drop('Species', axis=1).values
    y_text = df['Species'].values
    
    # แปลง Label (Species) ให้เป็นตัวเลข (0, 1, 2)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    
    # แบ่งข้อมูลเป็น Training 80% และ Testing 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ทำ Standardization ปรับสเกลข้อมูล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # แปลงมิติข้อมูลสำหรับ 1D CNN: (จำนวนตัวอย่าง, จำนวนฟีเจอร์, 1)
    X_train_reshaped = np.expand_dims(X_train_scaled, axis=2)
    X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, encoder.classes_