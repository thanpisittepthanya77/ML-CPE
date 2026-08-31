# นำเข้าฟังก์ชันจากไฟล์ที่เราเขียนแยกไว้
from data_loader import load_and_split_data
from evaluate import preprocess_data, train_and_evaluate_knn

def main():
    # 1. ระบุพาร์ทไปยังไฟล์ CSV 
    # (ใช้ ../ เพื่อถอยออกจากโฟลเดอร์ classification ไปหาโฟลเดอร์ data-animal)
    filepath = "data-animal/Iris.csv"
    
    # 2. โหลดและแยกข้อมูล (เรียกใช้จาก data_loader.py)
    X, y = load_and_split_data(filepath)
    
    # 3. เตรียมข้อมูลและปรับสเกล (เรียกใช้จาก evaluate.py)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 4. เทรนและหาค่า K ที่ดีที่สุด (เรียกใช้จาก evaluate.py)
    train_and_evaluate_knn(X_train, X_test, y_train, y_test, k_values=[3, 5, 7])

if __name__ == "__main__":
    main()