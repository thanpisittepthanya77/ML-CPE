from ultralytics import YOLO

def run_prediction():
    print("--- เริ่มต้นการทดสอบตรวจจับวัตถุ (Inference) ---")
    
    # โหลดโมเดลที่ดีที่สุด
    model_path = 'runs/detect/LAB10_Results/Config_A_Epoch10-3/weights/best.pt'
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"ไม่พบโมเดล โปรดรัน train.py ก่อนครับ")
        return

    # 1. ทดสอบกับรูปภาพ (เตรียมไฟล์ test_image.jpg ไว้ในโฟลเดอร์โปรเจกต์)
    print("\nทดสอบกับรูปภาพ...")
    model.predict(source='test_image.jpg', save=True, show=True)

    # 2. ทดสอบกับวิดีโอ (เอาคอมเมนต์ออกหากมีไฟล์วิดีโอ)
    # print("\nทดสอบกับวิดีโอ...")
    # model.predict(source='test_video.mp4', save=True, show=True)

    # 3. ทดสอบกับกล้องแบบเรียลไทม์ (กด 'q' เพื่อออก)
    # print("\nทดสอบกับกล้อง Webcam...")
    # model.predict(source='0', show=True)

if __name__ == '__main__':
    run_prediction()