from ultralytics import YOLO

def run_evaluation():
    print("--- เริ่มต้นการประเมินผลโมเดล (Evaluation) ---")
    
    # โหลดโมเดลตัวที่ดีที่สุดที่เพิ่งเทรนเสร็จ
    model_path = 'runs/detect/LAB10_Results/Config_A_Epoch10-3/weights/best.pt'
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"ไม่พบไฟล์โมเดลที่ {model_path} โปรดรัน train.py ก่อนครับ")
        return

    # คำนวณค่า Metrics ต่างๆ
    metrics = model.val()
    
    print("\n--- ผลการประเมิน (Performance Metrics) ---")
    print(f"mAP50-95 (ความแม่นยำรวม): {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__ == '__main__':
    run_evaluation()