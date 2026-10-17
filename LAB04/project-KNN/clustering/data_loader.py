import pandas as pd

def load_data_for_clustering(filepath):
    # โหลดไฟล์ CSV
    df = pd.read_csv(filepath)
    
    # ลบคอลัมน์ Id ทิ้ง
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    print("--- 1. Data Loading (Clustering) ---")
    print(f"โหลดข้อมูลทั้งหมด: {df.shape[0]} แถว")
    
    # แยก Feature (X) ออกมาโดยไม่ให้โมเดลเห็นเฉลย
    target_col = df.columns[-1] 
    X = df.drop(target_col, axis=1)
    actual_y = df[target_col] # เก็บเฉลยไว้เทียบทีหลัง
    
    return X, actual_y