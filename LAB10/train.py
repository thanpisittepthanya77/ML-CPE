from ultralytics import YOLO

def run_training():
    print("--- เริ่มต้นการฝึกสอนโมเดล (Training) ---")
    
    # โหลดโมเดล YOLOv8 ขนาดเล็ก (nano) ที่ผ่านการพรีเทรนมาแล้ว
    model = YOLO('yolov8n.pt') 
    
    # เริ่มเทรนโมเดล
    # สามารถปรับค่า epochs และ batch เพื่อเปรียบเทียบผลลัพธ์ตามใบงานได้
    model.train(
        data='coco8.yaml', 
        epochs=10,                # จำนวนรอบที่ต้องการเทรน
        batch=8,                  # ขนาดข้อมูลที่ประมวลผลต่อครั้ง
        imgsz=480,                # ขนาดรูปภาพ
        project='LAB10_Results',  # โฟลเดอร์หลักสำหรับเก็บผลลัพธ์
        name='Config_A_Epoch10'   # ชื่อโฟลเดอร์ย่อยของการรันครั้งนี้
    )
    print("การฝึกสอนเสร็จสิ้น! โมเดลถูกบันทึกไว้ใน LAB10_Results/Config_A_Epoch10/weights/best.pt")

if __name__ == '__main__':
    run_training()