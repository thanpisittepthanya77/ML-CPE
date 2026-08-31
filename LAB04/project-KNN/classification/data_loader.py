import pandas as pd

def load_and_split_data(filepath):
    """
    โหลดข้อมูลจากไฟล์ CSV และแยก Feature (X) กับ Target (y)
    """
    # โหลดไฟล์ CSV
    df = pd.read_csv(filepath)
    
    # ลบคอลัมน์ Id ทิ้งถ้ามี (เพราะเป็นแค่ตัวเลขลำดับ ไม่ใช้ทำนาย)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    print("--- 1. Data Loading ---")
    print(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    print("ตัวอย่างข้อมูล 5 แถวแรก:")
    print(df.head())
    
    # แยกระหว่าง ข้อมูลที่ใช้ทำนาย (X) และ คำตอบ (y)
    # สมมติว่าคอลัมน์คำตอบอยู่ขวาสุด (คอลัมน์สุดท้าย)
    target_col = df.columns[-1] 
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y