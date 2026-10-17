from data_loader import load_and_preprocess
from nn_model import build_and_train_nn
from evaluate import evaluate_and_save

def main():
    # ดึงไฟล์ข้อมูล Iris จาก LAB04 มาใช้งานได้เลย
    filepath = "LAB04/data-animal/Iris.csv" 
    
    print("1. กำลังเตรียมข้อมูล...")
    X_train, X_test, y_train, y_test, encoder = load_and_preprocess(filepath)
    
    print("2. กำลังสร้างและฝึกสอน Neural Network...")
    model = build_and_train_nn(X_train, y_train)
    
    print("3. กำลังประเมินผลและสรุปกราฟ...")
    evaluate_and_save(model, X_test, y_test)

if __name__ == "__main__":
    main()