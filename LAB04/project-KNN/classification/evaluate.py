from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def preprocess_data(X, y):
    """
    แบ่งข้อมูล Train/Test และทำ Standardization (ปรับสเกล)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k_values=[3, 5, 7]):
    """
    เทรนโมเดล KNN ตามค่า k ที่กำหนด และหาค่า k ที่ดีที่สุด
    """
    print("\n--- 2. KNN Training & Evaluation ---")
    accuracy_results = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracy_results[k] = acc
        print(f"Accuracy for k={k}: {acc:.4f} (หรือ {acc*100:.2f}%)")
        
    best_k = max(accuracy_results, key=accuracy_results.get)
    print(f"\n[Result] The best k value is: {best_k} with accuracy: {accuracy_results[best_k]:.4f}")
    
    return accuracy_results, best_k