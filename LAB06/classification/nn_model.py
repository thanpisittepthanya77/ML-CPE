from sklearn.neural_network import MLPClassifier

def build_and_train_nn(X_train, y_train):
    # สร้าง Neural Network โดยกำหนด Hidden Layer 2 ชั้น (ชั้นละ 10 neurons) และใช้ ReLU[cite: 2]
    # กำหนด Optimizer (solver) เป็น sgd ตามสไลด์[cite: 2]
    model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', 
                          solver='sgd', learning_rate_init=0.05, 
                          max_iter=500, random_state=42)
    
    print("เริ่มฝึกสอนโมเดล (Training)...")
    model.fit(X_train, y_train)
    
    return model