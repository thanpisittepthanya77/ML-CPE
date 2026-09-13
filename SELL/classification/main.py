from data_prep import get_prepared_data
from model_dev import train_svm_model, evaluate_and_plot  # แก้ไขบรรทัดนี้

def main():
    print("1. Loading and Preprocessing Data...")
# ... (โค้ดส่วนที่เหลือเหมือนเดิม)

def main():
    print("1. Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test, target_names = get_prepared_data()
    
    print("2. Training SVM Model...")
    svm_model = train_svm_model(X_train, y_train)
    
    print("3. Evaluating Model...\n")
    evaluate_and_plot(svm_model, X_test, y_test, target_names)

if __name__ == "__main__":
    main()