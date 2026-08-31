from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocess_and_cluster(X, n_clusters=3):
    print("\n--- 2. Data Preprocessing & K-Means Clustering ---")
    # 1. ทำ Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. สร้างและเทรนโมเดล K-Means (แบ่งเป็น 3 กลุ่มตามสายพันธุ์ไอริส)
    # ใช้ n_init="auto" เพื่อป้องกัน Warning ในเวอร์ชันใหม่
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    
    # ดึงผลลัพธ์ว่าแต่ละแถวถูกจัดไปอยู่กลุ่มไหน (0, 1 หรือ 2)
    predicted_clusters = kmeans.labels_
    
    print(f"จัดกลุ่มเสร็จสิ้น! โมเดลแบ่งข้อมูลออกเป็น {n_clusters} กลุ่ม")
    print("ตัวอย่างผลการจัดกลุ่ม 15 ตัวแรก:", predicted_clusters[:15])
    
    return predicted_clusters