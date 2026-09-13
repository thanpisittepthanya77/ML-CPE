from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_svm_model(X_train, y_train):
    # สร้างและฝึกสอนโมเดล SVM (Linear Kernel)
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_and_plot(model, X_test, y_test, target_names):
    # ทำนายผล
    y_pred = model.predict(X_test)
    
    # แสดงผลลัพธ์ตัวเลข
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # วาด Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - SVM Iris')
    plt.show()