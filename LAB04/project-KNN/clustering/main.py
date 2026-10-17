from data_loader import load_data_for_clustering
from evaluate import preprocess_and_cluster

def main():
    # อ้างอิง Path เดียวกับที่ทำใน Classification
    filepath = "data-animal/Iris.csv" 
    
    # 1. โหลดข้อมูล
    X, actual_y = load_data_for_clustering(filepath)
    
    # 2. ปรับสเกลและจัดกลุ่มด้วย K-Means
    labels = preprocess_and_cluster(X, n_clusters=3)

if __name__ == "__main__":
    main()