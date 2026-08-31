from data_loader import load_and_preprocess
from svm_models import train_svm_kernels
from evaluate import evaluate_and_save_outputs
filepath = "LAB04/data-animal/Iris.csv"
def main():
    # อ้างอิงโฟลเดอร์ data ที่อยู่ถอยออกไป 1 ขั้น
    filepath = "LAB04/data-animal/Iris.csv"
    
    print("1. กำลังโหลดและปรับสเกลข้อมูล (Standardization)...")
    X_train_scaled, X_test_scaled, y_train, y_test, X_test_original = load_and_preprocess(filepath)
    
    print("2. กำลังเทรนโมเดล SVM ด้วย 3 เคอร์เนล (Linear, Polynomial, RBF)...")
    models = train_svm_kernels(X_train_scaled, y_train)
    
    print("3. กำลังประเมินผลและสร้างไฟล์ Output...")
    evaluate_and_save_outputs(models, X_test_scaled, y_test, X_test_original)

if __name__ == "__main__":
    main()