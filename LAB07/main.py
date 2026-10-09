# main.py
from data_prep import load_and_prep_data
from model import build_cnn_model
from evaluate import plot_history, evaluate_model

def main():
    # 1. โหลดและเตรียมข้อมูล 
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, classes = load_and_prep_data("Iris.csv")
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    
    # 2. เปรียบเทียบการตั้งค่า (Configurations)
    configs = [
        {"name": "Config A (16 Filters, 50 Epochs)", "filters": 16, "epochs": 50},
        {"name": "Config B (32 Filters, 100 Epochs)", "filters": 32, "epochs": 100}
    ]
    
    for config in configs:
        print(f"\n--- Training {config['name']} ---")
        # สร้างโมเดล
        model = build_cnn_model(input_shape=input_shape, num_classes=3, filters=config["filters"])
        
        # เทรนโมเดล
        history = model.fit(
            X_train, y_train,
            epochs=config["epochs"],
            batch_size=8,
            validation_split=0.2, 
            verbose=1
        )
        
        # ส่ง classes เข้าไปในฟังก์ชันนี้ตามที่แก้ไว้
        evaluate_model(model, X_test, y_test, classes, config_name=config["name"])
        evaluate_model(model, X_test, y_test, classes, config_name=config["name"])
        # แสดงผลกราฟ
        plot_history(history, title_suffix=f"({config['name']})")
        
    print("\nTraining and Evaluation Completed!")

if __name__ == "__main__":
    main()