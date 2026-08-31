import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    # โหลดชุดข้อมูล
    df = pd.read_csv(filepath)
    
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ปรับสเกลข้อมูลให้เป็นมาตรฐานเดียวกันก่อนเทรนโมเดล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test