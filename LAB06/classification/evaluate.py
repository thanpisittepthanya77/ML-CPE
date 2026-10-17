import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def evaluate_and_save(model, X_test, y_test):
    # กำหนดให้สร้างโฟลเดอร์ outputs ไว้ใน LAB06/classification
    output_dir = "LAB06/classification/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # ทำนายและวัดความแม่นยำ
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- ผลการประเมินโมเดล Neural Network ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # วาดกราฟ Loss Curve (ดูค่า Error ที่ลดลงในแต่ละ Epoch)
    plt.figure(figsize=(6, 5))
    plt.plot(model.loss_curve_, label='Training Loss', color='red')
    plt.title('Neural Network Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_loss.png')
    plt.close()
    print(f"บันทึกกราฟประวัติการเทรนลงใน '{output_dir}/training_loss.png' สำเร็จ!")