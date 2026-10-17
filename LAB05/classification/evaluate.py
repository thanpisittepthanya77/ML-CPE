import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_and_save_outputs(models, X_test_scaled, y_test, X_test_original):
    # สร้างโฟลเดอร์ outputs ไว้ใน LAB05/classification
    output_dir = "LAB05/classification/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = X_test_original.copy()
    results_df['Actual_Class'] = y_test.values
    
    accuracies = []
    kernels = list(models.keys())
    
    print("\n--- SVM Kernel Performance ---")
    
    for k, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        # 1. ประเมินและแสดงคะแนน Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy ({k.capitalize()} kernel): {acc:.4f}")
        
        # เก็บผลการทำนายลงตาราง
        results_df[f'Predicted_{k}'] = y_pred
        
        # 2. วาดกราฟ Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix (Kernel = {k.capitalize()})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{output_dir}/01_confusion_matrix_{k}.png')
        plt.close()

    # 3. วาดกราฟแท่งเปรียบเทียบความแม่นยำ
    plt.figure(figsize=(8, 5))
    plt.bar(kernels, accuracies, color=['#4C72B0', '#DD8452', '#55A868'])
    plt.title('SVM Kernels Accuracy Comparison')
    plt.xlabel('Kernel Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig(f'{output_dir}/02_accuracy_comparison.png')
    plt.close()

    # 4. บันทึกผลการทำนายทั้งหมดลงไฟล์ CSV
    results_df.to_csv(f'{output_dir}/predictions.csv', index=False)
    print(f"\nบันทึกไฟล์ Outputs (กราฟและ CSV) ลงใน '{output_dir}' สำเร็จ!")